import matplotlib.pyplot as plt
import numpy as np
import cv2


def z_score(arr): 
    result = np.zeros(arr.shape)
    for i in range(arr.shape[2]):
        if arr[:,:,i].max() != 0 :
            mean = np.mean(arr[:,:,i])
            std = np.std(arr[:,:,i])
            result[:,:,i:i+1] = (arr[:,:,i:i+1] - mean) / std 
    return result

def gausian_norm(arr):
    for i in range(arr.shape[2]):
        if arr[:,:,i:i+1].min() != 0:
            arr[:,:,i:i+1] = arr[:,:,i:i+1] - arr[:,:,i:i+1].min()
            arr[:,:,i:i+1] = arr[:,:,i:i+1] / arr[:,:,i:i+1].max()
    return arr



########################
# preprocessing results
########################

def divide_cls(seg):
    TC = np.zeros(seg.shape)
    EC = np.zeros(seg.shape)
    WT = np.zeros(seg.shape)
    
    for D in range(seg.shape[2]):
        for H in range(seg.shape[1]):
            for W in range(seg.shape[0]):

                if seg[W,H,D] == 1:
                    TC[W,H,D] = 1
                    WT[W,H,D] = 1
                elif seg[W,H,D] == 2:
                    WT[W,H,D] = 1
                elif seg[W,H,D] == 4:
                    EC[W,H,D] = 1
                    TC[W,H,D] = 1
                    WT[W,H,D] = 1
    return TC, EC, WT


def divide_BG(seg, WT):
    BG = np.zeros(seg.shape)
    NT = np.zeros(seg.shape)
    
    for D in range(seg.shape[2]):
        for H in range(seg.shape[1]):
            for W in range(seg.shape[0]):
                if seg[W,H,D] > 0:
                    NT[W,H,D] = 1
                    
    BG = cv2.bitwise_not(NT, seg.shape)
    NT = NT - WT
    
    return BG, NT


def devide_col(ch2_img, ch3_img, ch4_img):
    
    ED = ch4_img - ch2_img
    NCR  = ch2_img - ch3_img
    ET = ch3_img
    return NCR, ET, ED

def color_type(NCR, ET, ED):
    RED = NCR+ET
    GREEN = ED+ET
    return RED, GREEN

def GT_color(R, G):
    add_img = np.concatenate((R, G, np.zeros((240,240,1))), axis = 2)
    return add_img

def convert_3d(raw_img):
    Con_img = np.concatenate((raw_img,raw_img,raw_img), axis = 2)
    return Con_img

def result_img(inp, TC, EC, WT):
    cls1, cls2, cls3 = devide_col(TC, EC, WT)
    Red , Green = color_type(cls1, cls2, cls3)
    mask_inv = np.reshape(cv2.bitwise_not(WT), WT.shape)
    other_img = np.reshape(cv2.bitwise_and(mask_inv, inp), np.shape(mask_inv))
    GT_img = GT_color(Red , Green)

    raw = convert_3d(other_img)
    OVR_img = raw + GT_img
    return OVR_img


def numpy_visual(Data):
    Dataset = Data
    cls = Dataset.shape[3]
    cnt = Dataset.shape[0]
    fig, ax = plt.subplots(cnt, cls, figsize = (15, 500))
    
    for i in range(cnt):
        for j in range(cls):
            
            ax[i,j].imshow(Dataset[i,:,:,j], cmap = 'gray')
            ax[i,j].axis("off")
            ax[i,j].set_title(i)