import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="3D Pix2Pix model for CBCT-to-CT translation")

    # Data paths and extensions
    parser.add_argument("-pathA", type=str, default="../croped_dataset/TRAINCBCTSIMULATED",
                        help="Path to the train data folder of domain A (source, e.g., CBCT)")
    parser.add_argument("-pathB", type=str, default="../croped_dataset/TRAINCTAlignedToCBCT",
                        help="Path to the train data folder of domain B (condition, e.g., CT)")
    parser.add_argument("-extensionA", type=str, default="nii",
                        help="Extension of domain A (source) data (nii default)")
    parser.add_argument("-extensionB", type=str, default="nii",
                        help="Extension of domain B (condition) data (nii default)")
    parser.add_argument("-splitterA", type=str, default="REC-",
                        help="Splitter to obtain volume ID for domain A. E.g., 'path/A-1.nii' -> splitter 'A-'")
    parser.add_argument("-splitterB", type=str, default="volume-",
                        help="Splitter to obtain volume ID for domain B. E.g., 'path/B-1.nii' -> splitter 'B-'")

    # Model architecture parameters
    parser.add_argument("-dimensions", nargs='+', type=int,default=[366,288,364],
                        help="Input shape of the model in rows, cols, depth. E.g.: -dimensions 256 256 64")
    parser.add_argument("--filterN", nargs='+', type=int, default=[32,64, 128, 256],
                        help="Amounts of filters per layer for the Generator. Needs to be 4 values. E.g.: -filterN 32 64 128 256")
    parser.add_argument('--kernalSize', type=int, default=3,
                        help='Size of (3D) convolutional filters. Default 3.')
    parser.add_argument("-downscale", action="store_true",
                        help="Apply initial downscaling (by factor 2) to input images. Useful for fitting data on GPU.")

    # Training parameters
    parser.add_argument('-learnRate', type=float, default=0.0002,
                        help='Learning rate for model training. Default 0.0002')
    parser.add_argument('-l2Regularization', type=float, default=0.2,
                        help='L2 based regularization for model training. Default 0.0002')
    parser.add_argument('-dropout', type=float, default=0.0, # Note: original TF was 0.2 for discriminator
                        help='Dropout rate for the discriminator. Default 0.0.')
    parser.add_argument('-epochs', type=int, default=10,
                        help='Amount of training epochs. Default 25')
    parser.add_argument('-batchSize', type=int, default=1,
                        help='Batch size. Default 1')
    parser.add_argument('-trainGenEach', type=int, default=1,
                        help='How often the Discriminator is trained for each Generator training step. Default 5.')
    parser.add_argument('-trivialThreshold', type=float, default=0.2,
                        help='Threshold to prevent Discriminator from converging to a trivial solution (all fake/real). Default 0.9.')
    parser.add_argument('-augRngThreshold', type=float, default=0.5, # Renamed for clarity as 'augRngThreshold'
                        help='Overall probability threshold (0.0-1.0) for applying any data augmentation to a sample. Default 0.5.')

    # Debugging and saving
    parser.add_argument('-sample_interval', type=int, default=50,
                    help='Interval to save sampled images. Default 50.')

    parser.add_argument('-debug', type=int, default=3,
                        help='Debug level. Higher value gives more debugging info. Default 3.')
    
    # Padding: Note on interpretation for nargs='+' with type=int
    # The help message implies 4 single integer values for padding (e.g., [4, 2, 2, 0])
    # If your CBCT2CTDataset expects a list of tuples like [(0,0),(0,0),(0,0),(0,0)],
    # you'll need to convert this list of ints to the correct format in your main script.
    parser.add_argument("-padding", nargs='+', type=int, default=[0, 0, 0, 0],
                        help="Padding values to apply to images: [pad_d, pad_h, pad_w, pad_c]. "
                             "E.g., for dimensions 180x118x182, padding 4 2 2 0 means symmetric padding "
                             "of 4, 2, 2 along D, H, W respectively, and 0 for channels.")
    
    parser.add_argument("-name", type=str, default='generator01',
                        help="Name for saved generator model files. Useful for automation scripts.")
    

    return parser.parse_args(args=[] if any("ipykernel" in arg or "-f=" in arg for arg in sys.argv) else None)


# Example of how you'd call this in your main training script:
# from args import parse_args
# args = parse_args()
# print(args.dimensions)

# !nvidia-smi

# !nvidia-smi



