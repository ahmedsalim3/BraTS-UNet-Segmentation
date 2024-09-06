import matplotlib.pyplot as plt
import numpy as np
import cv2


def z_score(data):
    """
    Compute the Z-score for each slice in the 3D array.
    
    Parameters:
    data (numpy.ndarray): A 3D NumPy array where the Z-score needs to be calculated for each slice along the third axis.
    
    Returns:
    numpy.ndarray: A 3D array with the Z-scores computed for each slice.
    """
    z_scores = np.zeros(data.shape)
    for i in range(data.shape[2]):
        if data[:,:,i].max() != 0 :
            mean = np.mean(data[:,:,i])
            std = np.std(data[:,:,i])
            z_scores[:,:,i:i+1] = (data[:,:,i:i+1] - mean) / std 
    return z_scores

def gaussian_norm(data):
    """
    Normalize each slice of a 3D array to a [0, 1] range using min-max normalization.

    Parameters:
    data (numpy.ndarray): A 3D NumPy array to be normalized.

    Returns:
    numpy.ndarray: A 3D array with normalized values in the [0, 1] range.
    """
    for i in range(data.shape[2]):
        if data[:,:,i:i+1].min() != 0:
            data[:,:,i:i+1] = data[:,:,i:i+1] - data[:,:,i:i+1].min()
            data[:,:,i:i+1] = data[:,:,i:i+1] / data[:,:,i:i+1].max()
    return data



########################
# preprocessing results
########################

def classify_segments(segmentation):
    """
    Classify and separate the segments of a 3D array into three categories based on their values.
    
    Parameters:
    segmentation (numpy.ndarray): A 3D NumPy array where each value represents a segment class.
    
    Returns:
    tuple: Three 3D NumPy arrays representing the classified segments:
           - `tumor_core`: Classifies segment value 1 and 4.
           - `enhance_core`: Classifies segment value 4.
           - `whole_tumor`: Classifies segment values 1, 2, and 4.
    """
    tumor_core = np.zeros(segmentation.shape)
    enhance_core = np.zeros(segmentation.shape)
    whole_tumor = np.zeros(segmentation.shape)
    
    for d in range(segmentation.shape[2]):
        for h in range(segmentation.shape[1]):
            for w in range(segmentation.shape[0]):

                if segmentation[w,h,d] == 1:
                    tumor_core[w,h,d] = 1
                    whole_tumor[w,h,d] = 1
                elif segmentation[w,h,d] == 2:
                    whole_tumor[w,h,d] = 1
                elif segmentation[w,h,d] == 4:
                    enhance_core[w,h,d] = 1
                    tumor_core[w,h,d] = 1
                    whole_tumor[w,h,d] = 1
    return tumor_core, enhance_core, whole_tumor


def classify_background_and_non_tumor(segmentation, whole_tumor):
    """
    Classify the background and non-tumor regions of a 3D array.

    Parameters:
    segmentation (numpy.ndarray): A 3D NumPy array with segmentation values.
    whole_tumor (numpy.ndarray): A 3D NumPy array with whole tumor regions.

    Returns:
    tuple: Two 3D NumPy arrays:
           - `background`: Classifies background regions.
           - `non_tumor`: Classifies non-tumor regions.
    """
    background = np.zeros(segmentation.shape)
    non_tumor = np.zeros(segmentation.shape)
    
    for d in range(segmentation.shape[2]):
        for h in range(segmentation.shape[1]):
            for w in range(segmentation.shape[0]):
                if segmentation[w,h,d] > 0:
                    non_tumor[w,h,d] = 1
                    
    background = cv2.bitwise_not(non_tumor , segmentation.shape)
    non_tumor  = non_tumor - whole_tumor
    
    return background, non_tumor


def separate_channels(img_ch2, img_ch3, img_ch4):
    ed = img_ch4 - img_ch2
    ncr = img_ch2 - img_ch3
    et = img_ch3
    return ncr, et, ed

def combine_colors(ncr, et, ed):
    red = ncr + et
    green = ed + et
    return red, green

def create_rgb_image(red, green):
    rgb_image = np.concatenate((red[..., np.newaxis], 
                                 green[..., np.newaxis], 
                                 np.zeros_like(red[..., np.newaxis])), 
                                axis=2)
    return rgb_image

def expand_to_rgb(raw_img):
    rgb_img = np.concatenate((raw_img[..., np.newaxis], 
                               raw_img[..., np.newaxis], 
                               raw_img[..., np.newaxis]), 
                              axis=2)
    return rgb_img


def results_image(input_img, tumor_core, enhance_core, whole_tumor):
    """
    Generate a result image by combining various channel images and applying masks.
    
    Parameters:
    input_img (numpy.ndarray): 2D input image.
    tumor_core (numpy.ndarray): Tumor core mask.
    enhance_core (numpy.ndarray): Enhanced core mask.
    whole_tumor (numpy.ndarray): Whole tumor mask.
    
    Returns:
    numpy.ndarray: 3D RGB image array with overlays.
    """
    cls1, cls2, cls3 = separate_channels(tumor_core, enhance_core, whole_tumor)
    red, green = combine_colors(cls1, cls2, cls3)
    
    mask_inv = np.reshape(cv2.bitwise_not(whole_tumor), whole_tumor.shape)
    masked_img = np.reshape(cv2.bitwise_and(mask_inv, input_img), mask_inv.shape)
    
    gt_img = create_rgb_image(red, green)
    raw_img_rgb = expand_to_rgb(masked_img)
    
    overlay_img = raw_img_rgb + gt_img
    return overlay_img


def visualize_data(data):
    """
    Visualize a 4D NumPy array as a grid of images.

    Parameters:
    data (numpy.ndarray): A 4D NumPy array with shape (num_images, height, width, num_channels).

    Returns:
    None: Displays the images in a grid using matplotlib.
    """
    num_images = data.shape[0]
    num_channels = data.shape[3]
    
    # Create a grid of subplots
    fig, ax = plt.subplots(num_images, num_channels, figsize=(15, 500))
    
    # Iterate over each image and channel
    for img_idx in range(num_images):
        for ch_idx in range(num_channels):
            ax[img_idx, ch_idx].imshow(data[img_idx, :, :, ch_idx], cmap='gray')
            ax[img_idx, ch_idx].axis("off")
            ax[img_idx, ch_idx].set_title(f'Image {img_idx}, Channel {ch_idx}')
    
    plt.tight_layout()
    plt.show()
