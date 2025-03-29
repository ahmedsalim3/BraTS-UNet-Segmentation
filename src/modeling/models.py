from tensorflow.keras.layers import (
    Input, SeparableConv2D, BatchNormalization, Activation, Conv2D, add, concatenate,
    MaxPool2D, multiply, Dropout
)

from tensorflow.keras.models import Model
from .blocks import Recurrent_block, up_conv


class DLUNetModel:
    """
    A class representing a Deep Learning U-Net Model with custom layers and blocks.
    """
    
    @staticmethod
    def Re_ASPP3(in_layer, ch, r):
        """
        Applies the Recurrent Atrous Spatial Pyramid Pooling (ASPP) block to the input layer.
        
        Args:
            in_layer (tensor): Input tensor.
            ch (int): Number of filters for convolution layers.
            r (int): Dilation rate for the convolutions.
        
        Returns:
            tensor: Output tensor after applying the ASPP block.
        """
        x1_1 = SeparableConv2D(filters=ch, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(in_layer)
        x1_1 = BatchNormalization()(x1_1)
        x1_1 = Activation('relu')(x1_1)
        
        x1_2 = Conv2D(filters=ch, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(x1_1)
        x1_2 = BatchNormalization()(x1_2)
        x1_2 = Activation('relu')(x1_2)
        
        x1_2 = add([x1_2, x1_1])

        # Repeating this process for the rest of the layers
        x2_1 = SeparableConv2D(filters=ch, kernel_size=(3, 3), padding='same', dilation_rate=r, kernel_initializer='he_normal')(in_layer)
        x2_1 = BatchNormalization()(x2_1)
        x2_1 = Activation('relu')(x2_1)
        
        x2_2 = Conv2D(filters=ch, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(x2_1)
        x2_2 = BatchNormalization()(x2_2)
        x2_2 = Activation('relu')(x2_2)
        
        x2_2 = add([x2_2, x2_1])
        
        x3_1 = Conv2D(filters=ch, kernel_size=(3, 3), padding='same', dilation_rate=r*2, kernel_initializer='he_normal')(in_layer)
        x3_1 = BatchNormalization()(x3_1)
        x3_1 = Activation('relu')(x3_1)

        x3_2 = Conv2D(filters=ch, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(x3_1)
        x3_2 = BatchNormalization()(x3_2)
        x3_2 = Activation('relu')(x3_2)

        x3_2 = add([x3_2, x3_1])
        
        x4_1 = Conv2D(filters=ch, kernel_size=(3, 3), padding='same', dilation_rate=r*3, kernel_initializer='he_normal')(in_layer)
        x4_1 = BatchNormalization()(x4_1)
        x4_1 = Activation('relu')(x4_1)
        
        x4_2 = Conv2D(filters=ch, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(x4_1)
        x4_2 = BatchNormalization()(x4_2)
        x4_2 = Activation('relu')(x4_2)
        
        x4_2 = add([x4_2, x4_1])
            
        x = concatenate([x1_2, x2_2, x3_2, x4_2, in_layer], axis=-1)

        x = Conv2D(filters=ch, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        return x
    
    @staticmethod
    def Attention_block(x, g, F_init):
        """
        Applies an attention block to the given feature maps.
        
        Args:
            x (tensor): Feature maps from the encoder.
            g (tensor): Feature maps from the decoder.
            F_init (int): Number of filters for convolution layers in the attention mechanism.
        
        Returns:
            tensor: Output tensor after applying the attention block.
        """
        g1 = Conv2D(filters=F_init, kernel_size=1, use_bias=True)(g)
        g1 = BatchNormalization()(g1)
        
        x1 = Conv2D(filters=F_init, kernel_size=1, use_bias=True)(x)
        x1 = BatchNormalization()(x1)

        gx = add([g1, x1])
        gx = Activation('relu')(gx)
        
        psi = Conv2D(filters=F_init*2, kernel_size=1, use_bias=True)(gx)
        psi = BatchNormalization()(psi)
        psi = Activation('sigmoid')(psi)
        
        out = multiply([x, psi])
        return out
    
    @staticmethod
    def RRCNN_block(inp, ch_out, t=2):
        """
        Applies a Recurrent Residual CNN block to the input tensor.
        
        Args:
            inp (tensor): Input tensor.
            ch_out (int): Number of output filters for convolution layers.
            t (int): Number of recurrent iterations. Default is 2.
        
        Returns:
            tensor: Output tensor after applying the RRCNN block.
        """
        x  = Conv2D(filters=ch_out, kernel_size=1, strides=1, padding='valid', kernel_initializer='he_normal')(inp)
        x  = Dropout(0.2)(x)
        x1 = Recurrent_block(x, ch_out=ch_out, t=t)
        x1 = Recurrent_block(x1, ch_out=ch_out, t=t)
        x1 = add([x, x1])
        return x1
    
    def build_model(self, input_shape):
        """
        Constructs the U-Net model with ASPP, attention blocks, and RRCNN blocks.
        
        Args:
            input_shape (tuple): Shape of the input images.
        
        Returns:
            Model: Keras model ready for Compiling.
        """
        inp = Input(input_shape)
        x1 = self.Re_ASPP3(inp, 64, 3)
        
        x2 = MaxPool2D()(x1)
        x2 = self.Re_ASPP3(x2, 128, 3)
        
        x3 = MaxPool2D()(x2)
        x3 = self.Re_ASPP3(x3, 256, 3)
        
        x4 = MaxPool2D()(x3)
        x4 = self.Re_ASPP3(x4, 512, 3)
        
        x5 = MaxPool2D()(x4)
        x5 = self.Re_ASPP3(x5, 1024, 3)
        
        # decoding + concat path
        d5 = up_conv(x5, 512)
        x4 = self.Attention_block(x=x4, g=d5, F_init=256)
        d5 = concatenate([x4, d5], axis=-1)
        d5 = self.RRCNN_block(d5, 512)
        
        d4 = up_conv(d5, 256)
        x3 = self.Attention_block(x=x3, g=d4, F_init=128)
        d4 = concatenate([x3, d4], axis=-1)
        d4 = self.RRCNN_block(d4, 256)
        
        d3 = up_conv(d4, 128)
        x2 = self.Attention_block(x=x2, g=d3, F_init=64)
        d3 = concatenate([x2, d3], axis=-1)
        d3 = self.RRCNN_block(d3, 128)
        
        d2 = up_conv(d3, 64)
        x1 = self.Attention_block(x=x1, g=d2, F_init=32)
        d2 = concatenate([x1, d2], axis=-1)
        d2 = self.RRCNN_block(d2, 64)

        d1 = Conv2D(filters=5, kernel_size=1, activation='sigmoid')(d2)
        
        model = Model(inp, d1)
        
        return model
