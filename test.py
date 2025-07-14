import torch
import numpy as np
import SimpleITK as sitk
import math
import os
import argparse

# Import your corrected modules
from parser import parse_args
from dataloader import CBCT2CTDataset 
from model import Generator # Assuming Generator is in models.py

def main():
    # --- Argument Parsing ---
    # Use the unified parse_args from args.py
    # Add pathGenerator and pathOutput to args.py if not already there, 
    # or define a separate parser for inference.
    # For simplicity, let's assume they are added to args.py for now.
    parser = argparse.ArgumentParser(description="3D Pix2Pix Transfer Function - PyTorch")
    parser.add_argument("-pathB", type=str, default="../croped_dataset/TESTCBCTSTIMULATED",
                        help="Path to the data folder of domain B (condition for generation)")
    parser.add_argument("-extensionB", type=str, default="nii",
                        help="Extension of domain B data (e.g., nii)")
    parser.add_argument("-splitterB", type=str, default="REC-",
                        help="Splitter to obtain volume ID (e.g., 'B-')")
    parser.add_argument("-pathGenerator", type=str,default='model_5/generator01.pth',
                        help="Path to the saved Generator model (.pth file)")
    parser.add_argument("-pathOutput", type=str,default='model_5/Generated_Test_CT',
                        help="Path to the folder where output NIfTI files will be saved")

    args = parser.parse_args([])  # Empty list means no args passed from notebook CLI

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Output Directory ---
    os.makedirs(args.pathOutput, exist_ok=True)
    print(f"Output will be saved to: {args.pathOutput}")

    # --- Load Generator Model ---
    # Generator needs to be instantiated with its architecture before loading state_dict
    # Input/Output channels should be 1 for medical images (grayscale)
    input_channels = 1
    output_channels = 1 
    generator = Generator(
        input_channels=input_channels, 
        output_channels=output_channels,
    ).to(device)

    # Load the state_dict (weights)
    try:
        # If your model was saved using torch.save(model.state_dict(), path)
        generator.load_state_dict(torch.load(args.pathGenerator, map_location=device))
        print(f"Successfully loaded generator weights from {args.pathGenerator}")
    except Exception as e:
        # If model was saved as a whole model (e.g., torch.save(model, path)), 
        # then torch.load would return the entire model object.
        # This is less common for best practices as it ties to specific code structure.
        print(f"Could not load state_dict, attempting to load entire model: {e}")
        try:
            generator = torch.load(args.pathGenerator, map_location=device)
            print(f"Successfully loaded entire generator model from {args.pathGenerator}")
        except Exception as e_full:
            print(f"Error loading model: {e_full}")
            print("Please ensure pathGenerator points to a valid .pth file "
                  "and that the Generator class matches the saved model's architecture.")
            return # Exit if model cannot be loaded

    generator.eval() # Set generator to evaluation mode (important for BatchNorm, Dropout)

    # --- DataLoader Setup ---
    # Convert padding from list of ints to list of tuples for the Dataset
    # Assuming symmetric padding: [p_d, p_h, p_w, p_c] -> [(p_d,p_d), (p_h,p_h), (p_w,p_w), (p_c,p_c)]
    # However, your removal logic `img[math.floor(padding[0]):,...]` implies padding was only at the start.
    # Let's align padding application with its removal here.
    # If padding is [D, H, W, C], then `np.pad(image, [(D,0),(H,0),(W,0),(C,0)], ...)` for application.
    # But for now, let's assume the previous dataset.py logic for symmetric application.
    
    # Original TF loader used `padding=[(padding[0], 0), (padding[1], 0), (padding[2], 0), (padding[3], 0)]`
    # This implies padding only on the "start" side of each dimension, which simplifies removal.
    # Let's make `dataset_padding` in CBCT2CTDataset reflect this for inference.
    
    # The `CBCT2CTDataset` expects `padding=[(pad_dim0_before, pad_dim0_after), ...]`.
    # So, if `args.padding = [d_pad, h_pad, w_pad, c_pad]` and you want to pad only at the start:

    # The dataloader is expecting data for domain A as input, but for inference,
    # we want to load images from `pathB` as inputs.
    # The `CBCT2CTDataset` loads two domains. For inference, we only need the 'condition' domain B.
    # We can pass `pathB` for both `pathA` and `pathB` to reuse the loader,
    # or modify `CBCT2CTDataset` to allow loading only one domain.
    # For now, let's load `pathB` as `pathA` and pass `rngThreshold=0` for no augmentation.
    # The dataloader returns `(imgsA, imgsB, sitkA, sitkB, indices)`.
    # Here `imgsA` will be the same as `imgsB` (from `pathB`). We will use `imgsB` for input.
    
    inference_dataset = CBCT2CTDataset(
        pathA=args.pathB,        # Load from pathB
        extensionA=args.extensionB,
        splitterA=args.splitterB,
        pathB=args.pathB,        # This will be the same as pathA for simplicity.
        extensionB=args.extensionB,
        splitterB=args.splitterB,
        rngThreshold=0.0,     # No augmentation for inference # Use the specific padding for inference
        augment=False,           # Explicitly disable augmentatio
    )

    inference_dataloader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False
    )
    print(f"Found {len(inference_dataset)} volumes for inference.")

    # --- Prediction Loop ---
    print("\nStarting prediction loop...")
    for batch_i, (imgs_A, imgs_B,indices) in enumerate(inference_dataloader):
        imgsB = imgs_B.to(device) # Move input to device

        with torch.no_grad(): # Disable gradient calculation for inference
            synthesized_batch = generator(imgsB).cpu().numpy() # Predict and move to CPU, convert to NumPy

        # Process each image in the batch
        for i, img_processed in enumerate(synthesized_batch):
            # img_processed shape: (C, D, H, W) - remove channel dimension
            img_processed = img_processed.squeeze(axis=0) # Remove channel dimension (e.g., from (1,D,H,W) to (D,H,W))

            save_path = os.path.join(args.pathOutput, f"{indices[i]}.nii")

            
            # Ensure the output image has correct dimensions and type for SimpleITK
            # SimpleITK expects (Z, Y, X) for 3D volumes
            # NumPy array should be (D, H, W) for medical images
            img_nifti = sitk.GetImageFromArray(img_processed)
            sitk.WriteImage(img_nifti, save_path)

            print(f"Saved: {save_path}")

    print("Done with all predictions!")



if __name__ == "__main__":
    main()


