# Import libraries 
import os
import numpy as np
import shutil
import math
import SimpleITK as sitk
import nibabel as nib
import os
from preprocessing_utils import gausian_norm, z_score


def n4_correction(source_dir, destination_dir):
    """
    - The `n4_correction` function applies N4 bias field correction to improve the uniformity of medical images.
    - Read images using SimpleITK
    - Apply Otsu thresholding to generate a mask
    - Perform N4 bias field correction
    - Save the corrected images to the destination directory
    """
    for folder1 in os.listdir(source_dir):   
        for folder2 in os.listdir(os.path.join(source_dir, folder1)):
            for file_name in os.listdir(os.path.join(source_dir, folder1, folder2)):
                file_path = source_dir + folder1 + '/' + folder2 + '/' + file_name
                target_file = destination_dir + folder1 + '/' + folder2 + '/'
                if 'seg.nii' in file_name:
                    if not os.path.exists(target_file):
                        os.makedirs(target_file)
                    shutil.copy(file_path, os.path.join(target_file, file_name))
                    
                elif 'ROI' in file_name:
                    continue
                else:
                    image = sitk.ReadImage(file_path)
                    mask = sitk.OtsuThreshold(image, 0, 1, 200)
                    image = sitk.Cast(image, sitk.sitkFloat32)
                    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
                    bias_corrector.SetNumberOfControlPoints([4, 4, 4])
                    bias_corrector.SetConvergenceThreshold(0.0001)
                    bias_corrector.SetMaximumNumberOfIterations([50, 40, 30])
                    corrected_image = bias_corrector.Execute(image, mask)
                    if not os.path.exists(target_file):
                        os.makedirs(target_file)
                    sitk.WriteImage(corrected_image, os.path.join(target_file, file_name))        
        


def crop_3d_images(source_dir, destination_dir, fix_size = 192):
    """
    - The `crop_3d_images` function crops 3D images to remove unnecessary background and focus on the region of interest.
    - Load 3D image volumes using `nibabel`
    - Identify the bounding box of non-zero intensity regions
    - Crop the image to the bounding box and pad to maintain a fixed size of 192x192x155
    - Save the cropped images
    
    """
    for folder1 in sorted(os.listdir(source_dir)):   
        for num, folder2 in enumerate(sorted(os.listdir(source_dir + folder1))):  
            send_to = destination_dir + folder1 + '/' + folder2 + '/' 
            for i, file_name in enumerate(sorted(os.listdir(source_dir + folder1 + '/' + folder2))):
                target_file = source_dir + folder1 + '/' + folder2 + '/' + file_name
                if not 'seg.nii' in target_file:
                    target = nib.load(target_file).get_fdata()
                    target_W = target.shape[0]
                    target_H = target.shape[1]
                    for W in range(target_W):
                        if np.max(target[W,:,:]) != 0:
                            x0 = W
                            break
                    for rW in range(target_W-1, 0, -1):
                        if np.max(target[rW,:,:]) != 0:
                            x1 = rW
                            break
                    for H in range(target_H):
                        if np.max(target[:,H,:]) != 0:
                            y0 = H
                            break
                    for rH in range(target_H-1, 0, -1):
                        if np.max(target[:,rH,:]) != 0:
                            y1 = rH
                            break
                    Cp_W = x1-x0
                    Cp_H = y1-y0
                    Background = np.zeros((fix_size, fix_size, 155))
                    W1 = (fix_size - Cp_W)/2
                    W2 = fix_size - W1
                    H1 = (fix_size - Cp_H)/2
                    H2 = fix_size - H1
                    Background[math.trunc(W1):math.trunc(W2), math.trunc(H1):math.trunc(H2), :] =  target[x0:x1, y0:y1, :]
                    if not os.path.exists(send_to):
                        os.makedirs(send_to)
                    np.save(send_to + file_name, Background)
                else:
                    target = nib.load(target_file).get_fdata()
                    seg = np.zeros((fix_size, fix_size, 155))
                    seg[math.trunc(W1):math.trunc(W2), math.trunc(H1):math.trunc(H2), :] =  target[x0:x1, y0:y1, :]
                    if not os.path.exists(send_to):
                        os.makedirs(send_to)
                    np.save(send_to + file_name, seg)
                    


def pre_processing(source_dir, destination_dir):
    for i,folder1 in enumerate(sorted(os.listdir(source_dir))):  
        for num, folder2 in enumerate(sorted(os.listdir(source_dir + folder1))):
            send_to = destination_dir + folder1 + '/' + folder2 + '/' 
            sand = np.zeros((155, 192, 192, 1))
            for j, filename in enumerate(sorted(os.listdir(source_dir + folder1 + '/' + folder2))):
                target_file = source_dir + folder1 + '/' + folder2 + '/' + filename
                if 'seg' in target_file:
                    seg = np.load(target_file)  
                    TC = np.zeros((seg.shape[2],seg.shape[0],seg.shape[1],1))
                    WT = np.zeros((seg.shape[2],seg.shape[0],seg.shape[1],1))
                    EC = np.zeros((seg.shape[2],seg.shape[0],seg.shape[1],1))
                    for D in range(seg.shape[2]):
                        for H in range(seg.shape[1]):
                            for W in range(seg.shape[0]):
                                
                                if seg[W,H,D] == 1:
                                    TC[D,W,H,0] = 1
                                    WT[D,W,H,0] = 1
                                elif seg[W,H,D] == 2:
                                    WT[D,W,H,0] = 1
                                elif seg[W,H,D] == 4:
                                    EC[D,W,H,0] = 1
                                    TC[D,W,H,0] = 1
                                    WT[D,W,H,0] = 1
                                    
                else:
                    inp = np.load(target_file)
                    data = z_score(inp)
                    data = gausian_norm(data)
                    data = np.transpose(data, (2,0,1))
                    #data = np.transpose(inp, (2,0,1))
                    data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], 1))
                    sand = np.concatenate([sand, data], axis = -1) 
            BG = np.zeros(TC.shape)
            NT = np.zeros(TC.shape)
            for D in range(seg.shape[2]):
                for H in range(seg.shape[1]):
                    for W in range(seg.shape[0]):
                        
                        if inp[W,H,D] == 0:
                            BG[D,W,H] = 1
                        elif inp[W,H,D] > 0:
                            NT[D,W,H] = 1
            
            NT = NT - WT
            GT = np.concatenate([BG, NT, TC, EC, WT], axis = -1)
            X_inp = sand[:,:,:,1:]
            if not os.path.exists(send_to):
                os.makedirs(send_to)            
            np.save(send_to + 'X.npy', X_inp)
            np.save(send_to + 'Y.npy', GT)
            # shutil.rmtree(source_dir + folder1 + '/' + folder2)
            


def create_dataset(source_dir, destination_dir):
    for i,folder1 in enumerate(sorted(os.listdir(source_dir))):  
        for j, folder2 in enumerate(sorted(os.listdir(source_dir + folder1))): 
            send_to = destination_dir + folder1 + '/' 
            for k, file_name in enumerate(sorted(os.listdir(source_dir + folder1 + '/' + folder2))):
                target_file = source_dir + folder1 + '/' + folder2 + '/' + file_name             
                if 'Y' in target_file:
                    Y = np.load(target_file)
                    X = np.load(source_dir + folder1 + '/' + folder2 + '/X.npy')
                    j = 0
                    for i in range(Y.shape[0]):
                        if np.max(Y[j,:,:,4:], initial = 0) == 0:
                            Y = np.delete(Y, (j), axis = 0)
                            X = np.delete(X, (j), axis = 0)
                        else:
                            j = j + 1
                    if not os.path.exists(send_to+str(j)+'/'):
                        os.makedirs(send_to+str(j)+'/')
                    np.save(send_to+str(j)+'/X.npy', X)
                    np.save(send_to+str(j)+'/Y.npy', Y)
                    # shutil.rmtree(source_dir + folder1 + '/' + folder2)
                    
                    
                    
def process_fold(source_dir, destination_dir, f = 5):
    j = 0 
    for i,folder1 in enumerate(sorted(os.listdir(source_dir))):   
        file_num  = len(os.listdir(source_dir + folder1))
        fold_num = round(file_num / f)
        for num, folder2 in enumerate(sorted(os.listdir(source_dir + folder1))):
            now_fold = math.trunc(num / fold_num)
            if now_fold == f:
                now_fold = now_fold - 1
            j = j + 1 
            Send_dir = destination_dir + str(now_fold) + '/'
            if not os.path.exists(Send_dir + str(j)):
                        os.makedirs(Send_dir + str(j))
            for k, file_name in enumerate(sorted(os.listdir(source_dir + folder1 + '/' + folder2))):
                target_file = source_dir + folder1 + '/' + folder2 + '/' + file_name
                shutil.copy(target_file, Send_dir + str(j) + '/'+ file_name)
                
                
def struct(source_dir, destination_dir):
    for i,folder1 in enumerate(sorted(os.listdir(source_dir))):
        X_concat = np.zeros((1,192,192,4))
        Y_concat = np.zeros((1,192,192,5))
        for j, folder2 in enumerate(sorted(os.listdir(source_dir + folder1))):
            for k, file_name in enumerate(sorted(os.listdir(source_dir + folder1 + '/' + folder2))):
                target_file = source_dir + folder1 + '/' + folder2 + '/' + file_name
                #                           1~5               1~N             X,Y
                if 'X' in file_name:
                    X = np.load(target_file)
                    X_concat = np.concatenate([X_concat, X], axis = 0)
                    # print(folder2)
                elif 'Y' in file_name:
                    Y = np.load(target_file)
                    Y_concat = np.concatenate([Y_concat, Y], axis = 0)
                    
        np.save(destination_dir + 'X_' + folder1+'.npy', X_concat[1:])
        np.save(destination_dir + 'Y_' + folder1+'.npy', Y_concat[1:])
        
