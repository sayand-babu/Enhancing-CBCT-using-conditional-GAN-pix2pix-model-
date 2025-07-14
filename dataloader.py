import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np
import torchio as tio
import os
from glob import glob


class CBCT2CTDataset(Dataset):
    def __init__(self, pathA, extensionA, splitterA, pathB, extensionB, splitterB, rngThreshold=0.5, padding=[(0,0),(0,0),(0,0),(0,0)],augment=False, downscale=False):
        # Initialize the parent Dataset class
        super().__init__()
        self.splitterA=splitterA
        self.extensionA=extensionA
        self.pathsA = sorted(glob(os.path.join(pathA, f'*.{extensionA}')),
                             key=lambda x: int(x.split(splitterA)[1].split(f'.{extensionA}')[0]))
        self.pathsB = sorted(glob(os.path.join(pathB, f'*.{extensionB}')),
                             key=lambda x: int(x.split(splitterB)[1].split(f'.{extensionB}')[0]))
        print(f'no of cbct samples :{len(self.pathsA)}')
        print(f'no of ct samples :{len(self.pathsB)}')
        
        self.rngThreshold = rngThreshold
        self.padding = padding
        self.augment = augment
        self.downscale = downscale
        effective_p = 1.0 - self.rngThreshold
        self.geometric_transforms = tio.Compose([
            tio.RandomFlip(axes=('LR',), p=effective_p),
            tio.RandomFlip(axes=('AP',), p=effective_p), 
            tio.RandomFlip(axes=('IS',), p=effective_p), 
        ])

        self.intensity_transforms = tio.Compose([
            tio.RandomBiasField(p=effective_p),
            tio.RandomBlur(p=effective_p),
            tio.RandomNoise(p=effective_p),
            tio.RandomGamma(p=effective_p)
        ])


    def __len__(self):
        return len(self.pathsA)

    def __getitem__(self, idx):
        # loading the sample at index idx 
        cbct_path = self.pathsA[idx]
        ct_path = self.pathsB[idx]
        cbct_sitk_image = sitk.ReadImage(cbct_path)
        ct_sitk_image = sitk.ReadImage(ct_path)
        cbct_np = sitk.GetArrayFromImage(cbct_sitk_image)
        ct_np = sitk.GetArrayFromImage(ct_sitk_image)

        #  Ensure 4D (D,H,W,C) after reading for consistency with process/TorchIO l
        if cbct_np.ndim == 3:
            cbct_np = np.expand_dims(cbct_np, axis=-1)
        if ct_np.ndim == 3:
            ct_np = np.expand_dims(ct_np, axis=-1)

        # Apply core preprocessing (Padding, Normalization, Downscaling)
        cbct_processed_np = self.process(cbct_np)
        ct_processed_np = self.process(ct_np)

        # Convert to PyTorch Tensors (channels-first) and float 
        cbct_tensor = torch.from_numpy(cbct_processed_np).permute(3, 0, 1, 2).float()
        ct_tensor = torch.from_numpy(ct_processed_np).permute(3, 0, 1, 2).float()

        # Apply augmentation directly to tensors (if self.augment is True)
        if self.augment and np.random.rand() < self.rngThreshold:
            cbct_tensor, ct_tensor = self.apply_augmentation(cbct_tensor, ct_tensor)
    
        return cbct_tensor, ct_tensor,idx

    def process(self, image):
        image = np.pad(image, self.padding, mode='constant', constant_values=0)                                                                       
        image = self.normalize(image)
        if self.downscale:
            image = image[::2, ::2, ::2]
        return image

    def normalize(self, data, min_std=1e-7):
        mean, std = data.mean(), data.std()
        return (data - mean) / std if std > min_std else data - mean

    def apply_augmentation(self, cbct_tensor, ct_tensor):
        subject = tio.Subject(
            cbct=tio.Image(tensor=cbct_tensor),
            ct=tio.Image(tensor=ct_tensor)
        )
        subject = self.geometric_transforms(subject)
        augmented_cbct_geo = subject['cbct'].data
        augmented_ct_geo = subject['ct'].data

        subject_cbct_only = tio.Subject(
            cbct=tio.Image(tensor=augmented_cbct_geo)
        )
        subject_cbct_only = self.intensity_transforms(subject_cbct_only)

        return subject_cbct_only['cbct'].data, augmented_ct_geo


