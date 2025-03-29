from tensorflow.keras.layers import (
    SeparableConv2D, BatchNormalization, Activation, add, UpSampling2D
)

# ----------------------
# MODELS BLOCKS
# ----------------------

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

# ----------------------
# Other Blocks (UNUSED)
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
