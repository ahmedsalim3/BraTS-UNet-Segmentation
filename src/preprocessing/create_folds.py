import os
import numpy as np
import shutil
import argparse
import math

def Create_Dataset(source_dir, destination_dir):
    """
    Process and filter 3D image datasets from the source directory and save the filtered datasets 
    to the destination directory. It removes samples where all values in the last few channels 
    of Y are zero, and then saves the filtered X and Y arrays.

    Args:
        source_dir (str): Directory containing the source images and segmentation files.
        destination_dir (str): Directory where the filtered datasets will be saved.
    """
    for folder1 in sorted(os.listdir(source_dir)):
        folder1_path = os.path.join(source_dir, folder1)
        if not os.path.isdir(folder1_path):
            continue
        for folder2 in sorted(os.listdir(folder1_path)):
            folder2_path = os.path.join(folder1_path, folder2)
            if not os.path.isdir(folder2_path):
                continue
            send_to = os.path.join(destination_dir, folder1)
            for file_name in sorted(os.listdir(folder2_path)):
                target_file = os.path.join(folder2_path, file_name)
                if not os.path.isfile(target_file):
                    print(f"File {target_file} does not exist, skipping.")
                    continue

                if 'Y' in file_name:
                    Y = np.load(target_file)
                    X = np.load(os.path.join(folder2_path, 'X.npy'))
                    j = 0
                    for i in range(Y.shape[0]):
                        if np.max(Y[j, :, :, 4:], initial=0) == 0:
                            Y = np.delete(Y, j, axis=0)
                            X = np.delete(X, j, axis=0)
                        else:
                            j += 1
                    send_to_folder = os.path.join(send_to, str(j))
                    if not os.path.exists(send_to_folder):
                        os.makedirs(send_to_folder)
                    np.save(os.path.join(send_to_folder, 'X.npy'), X)
                    np.save(os.path.join(send_to_folder, 'Y.npy'), Y)
                    # Optionally remove the source directory after processing
                    # shutil.rmtree(folder2_path)

def Process_Folds(source_dir, destination_dir, f=5):
    """
    Split the dataset into 'f' folds and save the files into corresponding fold directories.
    Each fold will contain approximately equal numbers of files.

    Args:
        source_dir (str): The source directory containing subdirectories with files to be split.
        destination_dir (str): The destination directory where the folds will be saved.
        f (int): The number of folds to create. Default is 5.
    """
    j = 0
    for folder1 in sorted(os.listdir(source_dir)):
        folder1_path = os.path.join(source_dir, folder1)
        if not os.path.isdir(folder1_path):
            continue
        file_num = len(os.listdir(folder1_path))
        fold_num = round(file_num / f)
        for num, folder2 in enumerate(sorted(os.listdir(folder1_path))):
            now_fold = math.trunc(num / fold_num)
            if now_fold == f:
                now_fold -= 1
            j += 1
            send_dir = os.path.join(destination_dir, str(now_fold))
            if not os.path.exists(os.path.join(send_dir, str(j))):
                os.makedirs(os.path.join(send_dir, str(j)))
            for file_name in sorted(os.listdir(os.path.join(folder1_path, folder2))):
                target_file = os.path.join(folder1_path, folder2, file_name)
                if os.path.isfile(target_file):
                    shutil.copy(target_file, os.path.join(send_dir, str(j), file_name))
                else:
                    print(f"File {target_file} does not exist, skipping.")

def Struct_Folds(source_dir, destination_dir):
    """
    Concatenate X and Y numpy files from subdirectories and save them as single numpy arrays.
    
    Args:
        source_dir (str): The source directory containing subdirectories with X and Y files.
        destination_dir (str): The destination directory where concatenated files will be saved.
    """
    for folder1 in sorted(os.listdir(source_dir)):
        folder1_path = os.path.join(source_dir, folder1)
        if not os.path.isdir(folder1_path):
            continue
        X_concat = np.zeros((1, 192, 192, 4))
        Y_concat = np.zeros((1, 192, 192, 5))
        for folder2 in sorted(os.listdir(folder1_path)):
            folder2_path = os.path.join(folder1_path, folder2)
            if not os.path.isdir(folder2_path):
                continue
            for file_name in sorted(os.listdir(folder2_path)):
                target_file = os.path.join(folder2_path, file_name)
                if not os.path.isfile(target_file):
                    print(f"File {target_file} does not exist, skipping.")
                    continue

                if 'X' in file_name:
                    X = np.load(target_file)
                    X_concat = np.concatenate([X_concat, X], axis=0)
                elif 'Y' in file_name:
                    Y = np.load(target_file)
                    Y_concat = np.concatenate([Y_concat, Y], axis=0)

        np.save(os.path.join(destination_dir, f'X_{folder1}.npy'), X_concat[1:])
        np.save(os.path.join(destination_dir, f'Y_{folder1}.npy'), Y_concat[1:])

if __name__ == "__main__":
    """The program's entry point."""
    parser = argparse.ArgumentParser(description='Process and create datasets from 3D images.')
    parser.add_argument(
        '--source-dir',
        type=str,
        required=True,
        help='Path to the source directory containing the original 3D images.'
    )
    parser.add_argument(
        '--destination-dir',
        type=str,
        required=True,
        help='Path to the destination directory for saving processed datasets.'
    )
    parser.add_argument(
        '--folds',
        type=int,
        default=5,
        help='Number of folds for splitting the dataset.'
    )
    args = parser.parse_args()

    Create_Dataset(args.source_dir, args.destination_dir)
    Process_Folds(args.source_dir, args.destination_dir, args.folds)
    Struct_Folds(args.source_dir, args.destination_dir)
