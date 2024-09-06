import os
import numpy as np
import nibabel as nib
import argparse
import math

def Crop_3D_Images(source_dir, destination_dir, size=192):
    """
    Crop each 3D image to remove empty or redundant regions around the brain,
    focusing only on the regions containing useful information. Default crop size is 192x192x155.

    Args:
        source_dir (str): Directory containing the source images and segmentation files.
        destination_dir (str): Directory where cropped files will be saved.
        size (int): Size to which images will be cropped. Default is 192.
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
            for file_name in sorted(os.listdir(folder2_path)):
                target_file = os.path.join(folder2_path, file_name)
                if not os.path.isfile(target_file):
                    print(f"File {target_file} does not exist, skipping.")
                    continue

                if 'seg.nii' in target_file:
                    target = nib.load(target_file).get_fdata()
                    seg = np.zeros((size, size, 155))
                    seg[math.trunc(W1):math.trunc(W2), math.trunc(H1):math.trunc(H2), :] = target[x0:x1, y0:y1, :]
                    np.save(os.path.join(send_to, file_name), seg)
                else:
                    target = nib.load(target_file).get_fdata()
                    target_W = target.shape[0]
                    target_H = target.shape[1]
                    
                    for W in range(target_W):
                        if np.max(target[W, :, :]) != 0:
                            x0 = W
                            break
                    for rW in range(target_W-1, 0, -1):
                        if np.max(target[rW, :, :]) != 0:
                            x1 = rW
                            break
                    for H in range(target_H):
                        if np.max(target[:, H, :]) != 0:
                            y0 = H
                            break
                    for rH in range(target_H-1, 0, -1):
                        if np.max(target[:, rH, :]) != 0:
                            y1 = rH
                            break
                            
                    Cp_W = x1 - x0
                    Cp_H = y1 - y0
                    Background = np.zeros((size, size, 155))
                    W1 = (size - Cp_W) / 2
                    W2 = size - W1
                    H1 = (size - Cp_H) / 2
                    H2 = size - H1
                    Background[math.trunc(W1):math.trunc(W2), math.trunc(H1):math.trunc(H2), :] = target[x0:x1, y0:y1, :]
                    np.save(os.path.join(send_to, file_name), Background)

if __name__ == "__main__":
    """The program's entry point."""
    parser = argparse.ArgumentParser(description='Crop 3D images to a fixed size.')
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
        help='Path to the destination directory where cropped images will be saved.'
    )
    parser.add_argument(
        '--fix-size',
        type=int,
        default=192,
        help='Fixed size for cropping (default: 192).'
    )
    args = parser.parse_args()

    Crop_3D_Images(args.source_dir, args.destination_dir, args.size)
