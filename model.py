import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# --- Helper Blocks ---

class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, groupnorm=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        layers = [nn.Conv3d(in_channels,filters, kernel_size, padding=padding, bias=not groupnorm)]
        if groupnorm:
            num_groups = min(8, filters // 2)
            layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size),
            ConvBlock(out_channels, out_channels, kernel_size)
        )

    def forward(self, x):
        return self.block(x)


# --- Generator (U-Net style) ---

# +

class Generator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, filters=32, kernel_size=3):
        super().__init__()
        self.enc1 = ConvBlock(input_channels, filters, kernel_size)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock(filters, filters * 2, kernel_size)
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock(filters * 2, filters * 4, kernel_size)

        self.up2 = nn.ConvTranspose3d(filters * 4, filters * 2, 2, stride=2)
        self.dec2 = ConvBlock(filters * 4, filters * 2, kernel_size)
        self.up1 = nn.ConvTranspose3d(filters * 2, filters, 2, stride=2)
        self.dec1 = ConvBlock(filters * 2, filters, kernel_size)

        self.final = nn.Conv3d(filters, output_channels, 1)

    def forward(self, x):
        # Use checkpointing on encoders and bottleneck
        e1 = checkpoint(self.enc1, x)
        p1 = self.pool1(e1)

        e2 = checkpoint(self.enc2, p1)
        p2 = self.pool2(e2)

        b = checkpoint(self.bottleneck, p2)

        u2 = self.up2(b)
        e2_cropped = self.center_crop(e2, u2.shape[2:])
        d2 = self.dec2(torch.cat([u2, e2_cropped], dim=1))

        u1 = self.up1(d2)
        e1_cropped = self.center_crop(e1, u1.shape[2:])
        d1 = self.dec1(torch.cat([u1, e1_cropped], dim=1))

        return self.final(d1)

    @staticmethod
    def center_crop(layer, target_spatial):
        _, _, d, h, w = layer.shape
        td, th, tw = target_spatial
        d1 = (d - td) // 2
        h1 = (h - th) // 2
        w1 = (w - tw) // 2
        return layer[:, :, d1:d1 + td, h1:h1 + th, w1:w1 + tw]


# -

# --- Discriminator ---

class Discriminator(nn.Module):
    def __init__(self, input_channels=2, filters=[32,64, 128, 256], filter_size=3, 
                 dropout_rate=0.0, batchnorm=True):
        super().__init__()

        self.conv1 = ConvBlock(input_channels, filters[0], filter_size)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = ConvBlock(filters[0], filters[1], filter_size)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = ConvBlock(filters[1], filters[2], filter_size)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4 = ConvBlock(filters[2], filters[3], filter_size)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.final_conv = nn.Conv3d(filters[3], 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        # Apply checkpointing only to deeper, memory-heavy blocks
        def conv3_block(inp):
            return self.conv3(self.pool2(inp))
        x = checkpoint(conv3_block, x)
        x = self.pool3(x)

        def conv4_block(inp):
            return self.conv4(self.pool3(inp))
        x = checkpoint(conv4_block, x)
        x = self.pool4(x)

        validity = self.final_conv(x)
        return validity



# --- GAN Wrapper (Optional but common) ---

# This class is a convenience wrapper to combine Generator and Discriminator for GAN training.
# As discussed, it's often more flexible to handle the training logic directly in the main loop,
# but this wrapper can be useful for defining the combined GAN forward pass.
# It makes sure the Discriminator's parameters are frozen when the Generator is being optimized.
class Vox3Vox(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

        # When initializing the GAN wrapper, freeze the discriminator's parameters.
        # This is for when you compute the GAN loss (Generator's objective).
        # You'll unfreeze it when training the discriminator.
        for param in self.discriminator.parameters():
            param.requires_grad = False

    def forward(self, imgA, imgB):
        # imgA: Real target image (e.g., real CT)
        # imgB: Condition input image (e.g., CBCT to translate from)

        # Generator creates a fake image A from condition B
        fakeA = self.generator(imgB)
        
        # Discriminator evaluates the fake pair (fakeA, imgB)
        # The discriminator's parameters are frozen during this forward pass if done within the GAN optimization step.
        validity = self.discriminator(fakeA, imgB)
        
        return validity, fakeA

# !nvidia-smi


