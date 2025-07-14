# -*- coding: utf-8 -*-
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

d_losses_history, g_total_losses_history, g_l1_losses_history = [], [], []
di_loss_real,di_loss_fake=[],[]

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

import nibabel as nib


import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from parser import parse_args
from dataloader import CBCT2CTDataset 
from torch.utils.data import DataLoader 
# Assuming your models (Generator, Discriminator) are defined in a file called 'models.py'
# and follow standard PyTorch nn.Module structure.
# Ensure your Generator and Discriminator take appropriate input_channels and filter counts.
# For 3D images, input_channels will likely be 1 for grayscale medical images.
from model import Generator, Discriminator 


# --- Helper Functions (moved for better organization) ---
def save_model(model, name):
    """Saves the state_dict of a PyTorch model."""
    # Create directory if it doesn't exist
    save_dir = "model_5"
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

def plot_losses(d_losses, g_total_losses, g_l1_losses, d_loss_fake_list, d_loss_real_list):
    """Plots and saves training loss curves with 3 subplots:
    1. Discriminator breakdown (real, fake, total)
    2. Generator breakdown (L1 and total)
    3. Combined overview of Discriminator & Generator total losses
    """

    import matplotlib.pyplot as plt
    import torch

    # --- Utility to convert any list of tensors to CPU floats ---
    def to_numpy_list(tensor_list):
        return [t.detach().cpu().item() if isinstance(t, torch.Tensor) else t for t in tensor_list]

    # --- Ensure all data is on CPU and converted to Python floats ---
    d_losses = to_numpy_list(d_losses)
    g_total_losses = to_numpy_list(g_total_losses)
    g_l1_losses = to_numpy_list(g_l1_losses)
    d_loss_fake_list = to_numpy_list(d_loss_fake_list)
    d_loss_real_list = to_numpy_list(d_loss_real_list)

    fig, axs = plt.subplots(3, 1, figsize=(12, 15))

    # --- Plot 1: Discriminator Loss Breakdown ---
    axs[0].plot(d_losses, label='Discriminator Total Loss', color='blue')
    axs[0].plot(d_loss_real_list, label='Discriminator Real Loss', color='orange')
    axs[0].plot(d_loss_fake_list, label='Discriminator Fake Loss', color='purple')
    axs[0].set_title('Discriminator Losses')
    axs[0].set_xlabel('Batch Iterations')
    axs[0].set_ylabel('Loss Value')
    axs[0].legend()
    axs[0].grid(True)

    # --- Plot 2: Generator Loss Breakdown ---
    axs[1].plot(g_total_losses, label='Generator Total Loss', color='green')
    axs[1].plot(g_l1_losses, label='Generator L1 Loss', color='red')
    axs[1].set_title('Generator Losses')
    axs[1].set_xlabel('Batch Iterations')
    axs[1].set_ylabel('Loss Value')
    axs[1].legend()
    axs[1].grid(True)

    # --- Plot 3: Combined Total Losses ---
    axs[2].plot(d_losses, label='Discriminator Total Loss', color='blue')
    axs[2].plot(g_total_losses, label='Generator Total Loss', color='green')
    axs[2].set_title('Combined Generator & Discriminator Total Losses')
    axs[2].set_xlabel('Batch Iterations')
    axs[2].set_ylabel('Loss Value')
    axs[2].legend()
    axs[2].grid(True)

    # --- Save and show ---
    plt.tight_layout()
    plt.savefig('model_5/training_loss_curves.png')
    plt.show()


def sample_images(real_A, cond_B, fake_A, epoch, batch_i, index):
    """
    Saves a central slice of real, generated, and original images for visualization.
    Inputs are PyTorch tensors on device.
    """
    # Pick middle slice for visualization along the depth dimension
    # Assuming input tensors are (B, C, D, H, W)
    idx = cond_B.shape[2] // 2 
    
    # Detach from graph, move to CPU, convert to NumPy
    real_A_np = real_A[0, 0, idx].detach().cpu().numpy() # Batch 0, Channel 0, Middle Depth
    fake_A_np = fake_A[0, 0, idx].detach().cpu().numpy()
    cond_B_np = cond_B[0, 0, idx].detach().cpu().numpy()

    images = [cond_B_np, fake_A_np, real_A_np] # Order: Condition (input to Gen), Generated (fake A), Original (real A)
    titles = ['Input (Domain B)', 'Generated (Domain A)', 'Ground Truth (Domain A)']

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        axs[i].imshow(images[i], cmap='gray', vmin=images[i].min(), vmax=images[i].max()) # Ensure proper colormap scaling
        axs[i].set_title(titles[i])
        axs[i].axis('off')

    # Create directory if it doesn't exist
    samples_dir = "model_5/sampled_images"
    os.makedirs(samples_dir, exist_ok=True)
    sample_path = os.path.join(samples_dir, f'image_sample_epoch{epoch:03d}_batch{batch_i:05d}_idx{index}.png')
    fig.savefig(sample_path)
    plt.close()
    if batch_i == 0: # Only print for the first sample to avoid too much console output
        print(f"Saved sample image to {sample_path}")


# +

def match_tensor_shape(source, target):
    """Crop or pad `source` to match shape of `target` along D, H, W axes."""
    _, _, d_s, h_s, w_s = source.shape
    _, _, d_t, h_t, w_t = target.shape

    # Padding if source is smaller
    pad_d = max(d_t - d_s, 0)
    pad_h = max(h_t - h_s, 0)
    pad_w = max(w_t - w_s, 0)

    # Pad in both directions (before, after)
    source = F.pad(source, (
        pad_w // 2, pad_w - pad_w // 2,
        pad_h // 2, pad_h - pad_h // 2,
        pad_d // 2, pad_d - pad_d // 2
    ))

    # Crop if source is larger
    _, _, d_s, h_s, w_s = source.shape
    d1 = (d_s - d_t) // 2
    h1 = (h_s - h_t) // 2
    w1 = (w_s - w_t) // 2

    return source[:, :, d1:d1 + d_t, h1:h1 + h_t, w1:w1 + w_t]


# -




# --- Main Training Function ---
def train_model(args): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Input shape for models (C, D, H, W)
    # Assuming single channel medical images
    input_channels = 1 
    model_input_shape = (input_channels, args.dimensions[0], args.dimensions[1], args.dimensions[2]) 
    
    # Discriminator patch shape for validity map
    # Output of discriminator is usually 1 channel (real/fake probability)
    discriminator_output_channels = 1 
    patch_dims = [int(x / (2 ** 4)) for x in args.dimensions] # Assuming 4 downsampling steps
    patch_shape = (discriminator_output_channels, patch_dims[0], patch_dims[1], patch_dims[2])
    print(f"Model input shape: {model_input_shape}")
    print(f"Discriminator patch shape: {patch_shape}")

    # --- DataLoader Initialization ---
    # Convert padding from list of ints to list of tuples for the Dataset
    # Assuming symmetric padding: [p_d, p_h, p_w, p_c] -> [(p_d,p_d), (p_h,p_h), (p_w,p_w), (p_c,p_c)]
    dataset_padding = [(p, p) for p in args.padding]
    print(f"Dataset padding configured as: {dataset_padding}")

    train_dataset = CBCT2CTDataset(
        pathA=args.pathA,
        extensionA=args.extensionA,
        splitterA=args.splitterA,
        pathB=args.pathB,
        extensionB=args.extensionB,
        splitterB=args.splitterB,
        rngThreshold=args.augRngThreshold, # Use the renamed argument
        padding=dataset_padding, # Pass the converted padding
        augment=True, # Enable augmentation for training
        downscale=args.downscale
    )

    # PyTorch DataLoader for efficient batching and shuffling
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batchSize,
        shuffle=True, 
        num_workers=0, # Use half cores, or 0 if single-core
        pin_memory=True if torch.cuda.is_available() else False # Pin memory for faster GPU transfer
    )
    print(f"Number of training samples: {len(train_dataset)}") 
    print(f"Number of batches per epoch: {len(train_dataloader)}")

    # --- Model Setup ---
    # Generator and Discriminator need to know input/output channels
    generator = Generator(
        input_channels=input_channels, 
        output_channels=input_channels, 
    ).to(device)

    discriminator = Discriminator(
        input_channels=input_channels * 2, # Concatenates A and B (e.g., 1+1=2 channels)
    ).to(device)

    # Loss functions
    criterion_GAN = nn.BCEWithLogitsLoss() # For GAN loss, stable for probabilities
    criterion_L1 = nn.L1Loss() # For L1 reconstruction loss

    # Optimizers
    # L2 regularization is usually handled by `weight_decay` in the optimizer
    optimizer_G = optim.Adam(generator.parameters(), lr=args.learnRate, betas=(0.5, 0.999)) # weight_decay=args.l2Regularization if applying to all params
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.learnRate, betas=(0.5, 0.999)) # weight_decay=args.l2Regularization if applying to all params

    # Store losses for plotting
    start_time = datetime.datetime.now()
    g=0
    d=0

    # --- Training Loop ---
    print("\nStarting training loop...")
    for epoch in range(args.epochs):
        for batch_i, (imgs_A, imgs_B,i,*_) in enumerate(train_dataloader):
            # Move images to the specified device (GPU if available)
            imgs_A = imgs_A.to(device) # Real A (target)
            imgs_B = imgs_B.to(device) # Condition B (input to generator)
            print('loaded image to gpu ')
            
            # downsampling 
           # imgs_A = torch.nn.functional.interpolate(imgs_A, scale_factor=0.25, mode='trilinear', align_corners=False)
           # imgs_B = torch.nn.functional.interpolate(imgs_B, scale_factor=0.25, mode='trilinear', align_corners=False)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Generate fake images
            d=d+1
            generator.eval()
            with autocast():
                fake_A = generator(imgs_B).detach()

                real_input_D = torch.cat((imgs_A, imgs_B), 1)
                fake_A = match_tensor_shape(fake_A, imgs_B)
                fake_input_D = torch.cat((fake_A, imgs_B), dim=1)

                real_validity = discriminator(real_input_D)
                fake_validity = discriminator(fake_input_D)
                         
                real_labels = torch.ones_like(real_validity)
                fake_labels = torch.zeros_like(fake_validity)

                d_loss_real = criterion_GAN(real_validity, real_labels)
                d_loss_fake = criterion_GAN(fake_validity, fake_labels)
                d_loss = 0.5 * (d_loss_real + d_loss_fake)

            scaler.scale(d_loss).backward()
            scaler.step(optimizer_D)
            scaler.update()
            torch.cuda.empty_cache()
            
            real_preds = (torch.sigmoid(real_validity) > 0.5).float()
            fake_preds = (torch.sigmoid(fake_validity) < 0.5).float() # Discriminator wants fake to be 0

            d_acc_real = torch.mean(real_preds).item() # How many real were correctly classified as real
            d_acc_fake = torch.mean(fake_preds).item() # How many fake were correctly classified as fake


            
            # -----------------
            #  Train Generator
            # -----------------
            
            if batch_i % args.trainGenEach == 0 and abs(d_acc_real - d_acc_fake) < 0.1:
                optimizer_G.zero_grad()

                generator.train() # Generator in training mode
                discriminator.eval() # Discriminator in evaluation mode for generator training step
                g=g+1
                # Generate fake images again (this time gradients are needed for generator)
                # ✅ DEBUG: Print input tensor info BEFORE passing to model
                print(f"Input shape: {imgs_B.shape}, device: {imgs_B.device}, dtype: {imgs_B.dtype}")
                print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
                with autocast():
                    fake_A = generator(imgs_B)
                    fake_A = match_tensor_shape(fake_A, imgs_B)
                    fake_input_D = torch.cat((fake_A, imgs_B), dim=1)
                    pred_fake = discriminator(fake_input_D)

                    g_loss_adv = criterion_GAN(pred_fake, real_labels)
                    g_loss_l1 = criterion_L1(fake_A, imgs_A)
                    g_loss = g_loss_adv + 20 * g_loss_l1
    
                scaler.scale(g_loss).backward()
                scaler.step(optimizer_G)
                scaler.update()
                torch.cuda.empty_cache()

                g_total_losses_history.append(g_loss.item())
                g_l1_losses_history.append(g_loss_l1.item()*20)
            else:
                g_loss = g_total_losses_history[-1] if g_total_losses_history else torch.tensor(0.0)
                g_loss_l1 = g_l1_losses_history[-1] if g_l1_losses_history else torch.tensor(0.0)
                g_loss = torch.tensor(g_loss, device=device)
                g_loss_l1 = torch.tensor(g_loss_l1, device=device)
                g_total_losses_history.append(g_loss)
                g_l1_losses_history.append(g_loss_l1)

            # Store losses
            d_losses_history.append(d_loss.item())
            di_loss_real.append(d_loss_real.item())
            di_loss_fake.append(d_loss_fake.item())

            # Print progress to console
            if args.debug > 0:
                elapsed_time = datetime.datetime.now() - start_time
                print(f"[Epoch {epoch+1}/{args.epochs}] [Batch {batch_i+1}/{len(train_dataloader)}] "
                      f"[D loss: {d_loss.item():.4f}] [G total loss: {g_loss.item():.4f}, G L1 loss: {g_l1_losses_history[-1]:.4f}] "
                      f"Time: {elapsed_time}")

            if batch_i % 25 == 0 and args.debug > 0:
                # Assuming batch_size is 1 for sampling ease. If larger, pick an index from batch.
                sample_images(imgs_A, imgs_B, fake_A, epoch, batch_i, 0) # Placeholder index 0

    # Save final model and plot losses after training
    save_model(generator, args.name)
    plot_losses(d_losses_history, g_total_losses_history, g_l1_losses_history,di_loss_fake,di_loss_real)

    print("Done training!")


if __name__ == "__main__":
    args = parse_args()
    train_model(args)



# !ps -u $USER | grep python

plt.plot([1, 2, 3])
plt.title("Test Plot")
plt.savefig("test_plot.png")








