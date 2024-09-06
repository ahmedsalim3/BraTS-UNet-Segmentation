from tensorflow.keras.layers import (
    Input, SeparableConv2D, BatchNormalization, Activation, Conv2D, add, concatenate,
    MaxPool2D, multiply, Dropout
)

from tensorflow.keras.models import Model
from utils import Recurrent_block, up_conv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import *


class DLUNetModel:
    
    @staticmethod
    def Re_ASPP3(in_layer, ch, r):
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
        x  = Conv2D(filters=ch_out, kernel_size=1, strides=1, padding='valid', kernel_initializer='he_normal')(inp)
        x  = Dropout(0.2)(x)
        x1 = Recurrent_block(x, ch_out=ch_out, t=t)
        x1 = Recurrent_block(x1, ch_out=ch_out, t=t)
        x1 = add([x, x1])
        return x1
    
    def build_model(self, input_shape):
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


# from model import DLUNetModel

# model_builder = DLUNetModel()
# model = model_builder.build_model((256, 256, 3))  # Example input shape
# model.summary()
  

class ImageVisualizer:
    def __init__(self, input_shape=(192, 192, 1)):
        """
        Initialize the ImageProcessor with the shape of input images.
        
        Args:
            input_shape (tuple): Shape of the input images. Default is (192, 192, 1).
        """
        self.input_shape = input_shape
    
    def divide_classes(self, ch2_img, ch3_img, ch4_img):
        """
        Compute the difference between channels to get NCR, ET, and ED.
        
        Args:
            ch2_img, ch3_img, ch4_img: Images corresponding to different channels.
            
        Returns:
            tuple: NCR, ET, ED images.
        """
        ed = ch4_img - ch2_img
        ncr = ch2_img - ch3_img
        et = ch3_img
        return ncr, et, ed

    def generate_color_layers(self, ncr, et, ed):
        """
        Generate RED and GREEN color layers from NCR, ET, and ED.
        
        Args:
            ncr, et, ed: Images corresponding to different classes.
            
        Returns:
            tuple: RED and GREEN composite images.
        """
        red = ncr + et
        green = ed + et
        return red, green

    def create_rgb_image(self, red_layer, green_layer):
        """
        Combine R and G layers with a zero B layer to create a colored image.
        
        Args:
            red_layer, green_layer: Images corresponding to red and green channels.
            
        Returns:
            ndarray: RGB image.
        """
        rgb_image = np.concatenate((red_layer, green_layer, np.zeros(self.input_shape)), axis=2)
        return rgb_image

    def convert_to_3d(self, image):
        """
        Convert a 1-channel image to a 3-channel image by replicating it.
        
        Args:
            image: Single-channel input image.
            
        Returns:
            ndarray: 3-channel image.
        """
        return np.concatenate((image, image, image), axis=2)

    def generate_result_image(self, input_img, mask_img):
        """
        Generate an overlaid result image with input and mask images.
        
        Args:
            input_img: Raw input image.
            mask_img: Mask image for comparison.
            
        Returns:
            ndarray: Combined overlay image.
        """
        ncr, et, ed = self.divide_classes(mask_img[..., 2:3], mask_img[..., 3:4], mask_img[..., 4:5])
        red, green = self.generate_color_layers(ncr, et, ed)
        
        mask_inv = np.reshape(cv2.bitwise_not(mask_img[..., 4:5]), np.shape(mask_img[..., 4:5]))
        other_img = np.reshape(cv2.bitwise_and(mask_inv, input_img[..., 0:1]), np.shape(mask_inv))
        
        gt_img = self.create_rgb_image(red, green)
        raw_img = self.convert_to_3d(other_img)
        
        overlaid_image = raw_img + gt_img
        return overlaid_image

    def arrange_images(self, input_img, mask_img, prediction):
        """
        Arrange images for display: Ground Truth, Prediction, and channel composites.
        
        Args:
            input_img: Raw input image.
            mask_img: Ground truth mask image.
            prediction: Model prediction mask image.
            
        Returns:
            tuple: GT image, Prediction image, TC, EC, and WT composites.
        """
        gt_image = self.generate_result_image(input_img[0], mask_img[0])
        pred_image = self.generate_result_image(input_img[0], prediction)
        
        tc = np.concatenate((prediction[..., 2:3], mask_img[0, :, :, 2:3], np.zeros(self.input_shape)), axis=2)
        ec = np.concatenate((prediction[..., 3:4], mask_img[0, :, :, 3:4], np.zeros(self.input_shape)), axis=2)
        wt = np.concatenate((prediction[..., 4:5], mask_img[0, :, :, 4:5], np.zeros(self.input_shape)), axis=2)
        
        return gt_image, pred_image, tc, ec, wt

    def visualize_results(self, test_images, test_masks, model):
        """
        Visualize the results for a set of test images and ground truth masks using a model.
        
        Args:
            test_images: Array of input test images.
            test_masks: Array of ground truth masks corresponding to the test images.
            model: Trained model to generate predictions.
        """
        for index in range(test_images.shape[0]):
            preds = np.squeeze(model.predict(test_images[index:index + 1], verbose=0))
            preds = (preds > 0.2).astype(np.float64)
            
            gt_img, pred_img, tc_img, ec_img, wt_img = self.arrange_images(test_images[index:index + 1], test_masks[index:index + 1], preds)
            
            tc_score, wt_score, ec_score = np.round(model.evaluate(x=test_images[index:index + 1], y=test_masks[index:index + 1])[3:6], 2)
            
            fig, axes = plt.subplots(1, 5, figsize=(20, 10))
            plt.imshow(pred_img)
            
            axes[0].imshow(gt_img)
            axes[0].set_title(f'GT : {index}', fontsize=15)
            axes[0].axis("off")
            
            axes[1].imshow(pred_img)
            axes[1].set_title('Prediction', fontsize=15)
            axes[1].axis("off")
            
            axes[2].imshow(tc_img, cmap='gray')
            axes[2].set_title(f'TC : {tc_score}', fontsize=15)
            axes[2].axis("off")
            
            axes[3].imshow(ec_img, cmap='gray')
            axes[3].set_title(f'EC : {ec_score}', fontsize=15)
            axes[3].axis("off")
            
            axes[4].imshow(wt_img, cmap='gray')
            axes[4].set_title(f'WT : {wt_score}', fontsize=15)
            axes[4].axis("off")


# Example Usage:
# visualizer = ImageVisualizer()
# visualizer.plot_all_views(X_test, Y_test, model)