import os
import SimpleITK as sitk
import shutil
import argparse

def N4BiasFieldCorrection(source_dir, destination_dir):
    """
    Applies N4 bias field correction to MRI images to address intensity non-uniformity 
    caused by artifacts from patient movement or scanner hardware, which can affect 
    neural network training.

    Args:
        source_dir (str): Directory containing MRI images
        destination_dir (str): Directory to save the corrected MRI images
    """
    for folder1 in os.listdir(source_dir):   
        folder1_path = os.path.join(source_dir, folder1)
        if not os.path.isdir(folder1_path):
            continue
        for folder2 in os.listdir(folder1_path):
            folder2_path = os.path.join(folder1_path, folder2)
            if not os.path.isdir(folder2_path):
                continue
            for file_name in os.listdir(folder2_path):
                file_path = os.path.join(folder2_path, file_name)
                target_folder = os.path.join(destination_dir, folder1, folder2)
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                
                if 'seg.nii' in file_name:
                    shutil.copy(file_path, os.path.join(target_folder, file_name))
                else:
                    image = sitk.ReadImage(file_path)
                    mask = sitk.OtsuThreshold(image, 0, 1, 200)
                    image = sitk.Cast(image, sitk.sitkFloat32)
                    
                    corrector = sitk.N4BiasFieldCorrectionImageFilter()
                    corrector.SetNumberOfControlPoints([4, 4, 4])
                    corrector.SetConvergenceThreshold(0.0001)
                    corrector.SetMaximumNumberOfIterations([50, 40, 30])
                    
                    img_corr = corrector.Execute(image, mask)
                    sitk.WriteImage(img_corr, os.path.join(target_folder, file_name))
                    
                    

if __name__ == "__main__":
    """The program's entry point."""
    parser = argparse.ArgumentParser(description='Run N4 bias field correction on MRI images.')
    parser.add_argument(
        '--source-dir',
        type=str,
        required=True,
        help='Path to the source directory containing MRI images.'
    )
    parser.add_argument(
        '--destination-dir',
        type=str,
        required=True,
        help='Path to the destination directory where corrected images will be saved.'
    )
    args = parser.parse_args()

    N4BiasFieldCorrection(args.source_dir, args.destination_dir)