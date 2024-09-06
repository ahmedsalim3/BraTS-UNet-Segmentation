from tensorflow.keras import backend as K
import tensorflow as tf 
from tensorflow.keras.layers import (
    SeparableConv2D, BatchNormalization, Activation, add, UpSampling2D
)


# ----------------------
# Models Utilities
# ----------------------

def up_conv(inp, ch_out):
    """
    Upsample and apply a separable convolution followed by batch normalization and ReLU activation.

    Args:
        inp (tensor): Input tensor.
        ch_out (int): Number of output channels.

    Returns:
        tensor: Processed tensor.
    """
    x = UpSampling2D()(inp)
    x = SeparableConv2D(filters=ch_out, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

################
# MODELS BLOCKS
################

def conv_one(inp, ch_out):
    """
    Apply a separable convolution followed by batch normalization and ReLU activation.

    Args:
        inp (tensor): Input tensor.
        ch_out (int): Number of output channels.

    Returns:
        tensor: Processed tensor.
    """
    x = SeparableConv2D(filters=ch_out, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def Recurrent_block(inp, ch_out, t= 2):
    """
    Apply a recurrent block of separable convolutions with residual connections.

    Args:
        inp (tensor): Input tensor.
        ch_out (int): Number of output channels.
        t (int): Number of recurrent steps.

    Returns:
        tensor: Processed tensor.
    """
    for i in range(t):
        if i == 0:
            x1 = conv_one(inp,ch_out)
        x1 = add([inp,x1])
        x1 = conv_one(x1,ch_out)
    return x1


# ----------------------
# Metric Utilities
# ----------------------

def channel_dice(c, smooth=1.0):
    """
    Compute the Dice coefficient for a specific channel.

    Args:
        c (int): Channel index.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        function: Dice coefficient metric function.
    """
    def dice_metric(y_true, y_pred):
        y_true_f = K.flatten(y_true[:, :, :, c])
        y_pred_f = K.flatten(y_pred[:, :, :, c])
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return dice

    dice_metric.__name__ = f'dice_metric_channel_{c}'
    return dice_metric

################
# CUSTOM METRIC
################

def dice_coef(y_true, y_pred, smooth=1):
    """
    Compute the Dice coefficient for binary or multi-class segmentation.

    Args:
        y_true (tensor): Ground truth labels.
        y_pred (tensor): Predicted labels.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        tensor: Dice coefficient.
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice

def mean_iou(y_true, y_pred, smooth=1):
    """
    Compute the mean Intersection over Union (IoU) for binary or multi-class segmentation.

    Args:
        y_true (tensor): Ground truth labels.
        y_pred (tensor): Predicted labels.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        tensor: Mean IoU.
    """

    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(y_pred > 0.5, "int32")
    
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
    union = K.sum(y_true,[1, 2])+K.sum(y_pred,[1,2])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=[1, 0])
    
    return iou 

################
# CUSTOM LOSS
################
def weighted_dice_loss(y_true, y_pred, weight=1):
    """
    Compute the weighted Dice loss.

    Args:
        y_true (tensor): Ground truth labels.
        y_pred (tensor): Predicted labels.
        weight (float): Weighting factor for Dice loss.

    Returns:
        tensor: Weighted Dice loss.
    """
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / (K.sum(w * (m1**2)) + K.sum(w * (m2**2)) + smooth) # Uptill here is Dice Loss with squared
    loss = 1. - K.sum(score)  #Soft Dice Loss
    return loss

def weighted_bce_loss(y_true, y_pred, weight):
    """
    Compute the weighted binary cross-entropy loss.

    Args:
        y_true (tensor): Ground truth labels.
        y_pred (tensor): Predicted labels.
        weight (tensor): Weighting factors.

    Returns:
        tensor: Weighted binary cross-entropy loss.
    """
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
        
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
    (K.log(1. + K.exp(-K.abs(logit_y_pred))) + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)

def weighted_bce_and_dice_loss(y_true, y_pred):
    """
    Compute a combination of weighted binary cross-entropy and Dice loss.

    Args:
        y_true (tensor): Ground truth labels.
        y_pred (tensor): Predicted labels.

    Returns:
        tensor: Combined loss.
    """
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
            y_true, pool_size=(11, 11), strides=(1, 1),padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2 
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    loss =  weighted_dice_loss(y_true, y_pred, weight) + weighted_bce_loss(y_true, y_pred, weight) 
    return loss


# ----------------------
# Other Utilities
# ----------------------

# from tensorflow.keras.layers import Conv2D, multiply
# def Attention_block(x, g, F_init):
#     g1 = Conv2D(filters=F_init, kernel_size=1, use_bias=True)(g)
#     g1 = BatchNormalization()(g1)
    
#     x1 = Conv2D(filters=F_init, kernel_size=1, use_bias=True)(x)
#     x1 = BatchNormalization()(x1)

#     gx = add([g1, x1])
#     gx = Activation('relu')(gx)
    
#     psi = Conv2D(filters=F_init*2, kernel_size=1, use_bias=True)(gx)
#     psi = BatchNormalization()(psi)
#     psi = Activation('sigmoid')(psi)
    
#     out = multiply([x, psi])
#     return out

# from tensorflow.keras.layers import Conv2D, Dropout
# def RRCNN_block(inp, ch_out, t=2):
#     x  = Conv2D(filters=ch_out, kernel_size=1, strides=1, padding='valid', kernel_initializer='he_normal')(inp)
#     x  = Dropout(0.2)(x)
#     x1 = Recurrent_block(x, ch_out=ch_out, t=t)
#     x1 = Recurrent_block(x1, ch_out=ch_out, t=t)
#     x1 = add([x, x1])
#     return x1

# from tensorflow.keras.layers import Dropout
# def Separable_block(inp, ch_out):
#     x= SeparableConv2D(filters=ch_out,kernel_size=3, padding='same', use_bias=True, kernel_initializer = 'he_normal')(inp)
#     x= BatchNormalization()(x)
#     x= Activation('relu')(x)
#     x= Dropout(0.2)(x) 
#     x= SeparableConv2D(filters=ch_out,kernel_size=3, padding='same', use_bias=True, kernel_initializer = 'he_normal')(x)
#     x= BatchNormalization()(x)
#     x= Activation('relu')(x)
#     x= Dropout(0.2)(x)
#     return x