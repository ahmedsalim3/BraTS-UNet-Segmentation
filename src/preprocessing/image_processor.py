import os
import numpy as np
import argparse
from .utils import gaussian_norm, z_score

def Process_Images(source_dir, destination_dir):
    """
    Process 3D images from the source directory and save the processed datasets to the destination directory.
    
    The function:
    - Normalizes images using z-score and Gaussian normalization techniques.
    - Pads and reshapes the images and masks.
    - Saves the preprocessed images as `X.npy` and combined masks (Background (BG), 
      Non-Tumor (NT), Tumor Core (TC), Enhancing Core (EC), Whole Tumor (WT)) as `Y.npy`.

    Args:
        source_dir (str): Directory containing the source images and segmentation files.
        destination_dir (str): Directory where processed files will be saved.
    """
    for folder1 in sorted(os.listdir(source_dir)):
        folder1_path = os.path.join(source_dir, folder1)
        if not os.path.isdir(folder1_path):
            continue
        for folder2 in sorted(os.listdir(folder1_path)):
            folder2_path = os.path.join(folder1_path, folder2)
            if not os.path.isdir(folder2_path):
                continue
            send_to = os.path.join(destination_dir, folder1, folder2)
            if not os.path.exists(send_to):
                os.makedirs(send_to)
                
            sand = np.zeros((155, 192, 192, 1))
            seg = None
            inp = None
            
            for filename in sorted(os.listdir(folder2_path)):
                target_file = os.path.join(folder2_path, filename)
                if not os.path.isfile(target_file):
                    print(f"File {target_file} does not exist, skipping.")
                    continue

                if 'seg' in filename:
                    seg = np.load(target_file)
                    TC = np.zeros((seg.shape[2], seg.shape[0], seg.shape[1], 1))
                    WT = np.zeros((seg.shape[2], seg.shape[0], seg.shape[1], 1))
                    EC = np.zeros((seg.shape[2], seg.shape[0], seg.shape[1], 1))
                    for D in range(seg.shape[2]):
                        for H in range(seg.shape[1]):
                            for W in range(seg.shape[0]):
                                if seg[W, H, D] == 1:
                                    TC[D, W, H, 0] = 1
                                    WT[D, W, H, 0] = 1
                                elif seg[W, H, D] == 2:
                                    WT[D, W, H, 0] = 1
                                elif seg[W, H, D] == 4:
                                    EC[D, W, H, 0] = 1
                                    TC[D, W, H, 0] = 1
                                    WT[D, W, H, 0] = 1
                else:
                    inp = np.load(target_file)
                    data = z_score(inp)
                    data = gaussian_norm(data)
                    data = np.transpose(data, (2, 0, 1))
                    data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))
                    sand = np.concatenate([sand, data], axis=-1)

            if seg is None or inp is None:
                print(f"No segmentation or input data found in {folder2_path}, skipping.")
                continue

            BG = np.zeros(TC.shape)
            NT = np.zeros(TC.shape)
            for D in range(seg.shape[2]):
                for H in range(seg.shape[1]):
                    for W in range(seg.shape[0]):
                        if inp[W, H, D] == 0:
                            BG[D, W, H] = 1
                        elif inp[W, H, D] > 0:
                            NT[D, W, H] = 1
            
            NT = NT - WT
            GT = np.concatenate([BG, NT, TC, EC, WT], axis=-1)
            X_inp = sand[:, :, :, 1:]
            np.save(os.path.join(send_to, 'X.npy'), X_inp)
            np.save(os.path.join(send_to, 'Y.npy'), GT)
            # Optionally remove the source directory after processing
            # shutil.rmtree(folder2_path)

if __name__ == "__main__":
    """The program's entry point."""
    parser = argparse.ArgumentParser(description='Preprocess 3D images and create datasets.')
    parser.add_argument(
        '--source-dir',
        type=str,
        required=True,
        help='Path to the source directory containing 3D images.'
    )
    parser.add_argument(
        '--destination-dir',
        type=str,
        required=True,
        help='Path to the destination directory where processed data will be saved.'
    )
    args = parser.parse_args()

    Process_Images(args.source_dir, args.destination_dir)
