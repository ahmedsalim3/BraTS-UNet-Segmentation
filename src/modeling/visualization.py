import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns



class ImageVisualizer:
    """
    A class for visualizing and processing images for model predictions and ground truth comparisons.
    """
    
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

class TrainingPlotter:
    @staticmethod
    def plot(history):
        sns.set_style("darkgrid")
        colors = {
            'train': '#13034d',
            'val': '#084d02'
        }
        
        epochs = range(1, len(history.history['loss']) + 1)
        
        metrics = {
            "Loss": ('loss', 'val_loss'),
            "Mean IoU": ('mean_iou', 'val_mean_iou'),
            "Dice Coefficient": ('dice_coef', 'val_dice_coef'),
            "Tumor Core Channel": ('channel_dice_2', 'val_channel_dice_2'),
            "Whole Tumor Channel": ('channel_dice_3', 'val_channel_dice_3'),
            "Enhancing Tumor Channel": ('channel_dice_4', 'val_channel_dice_4')
        }
        
        fig, axs = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
        
        def plot_metric(ax, train_data, val_data, title, ylabel, ylim=None):
            ax.plot(epochs, train_data, marker='o', markersize=6, linewidth=2, color=colors['train'], label="Training")
            ax.plot(epochs, val_data, marker='s', markersize=6, linewidth=2, color=colors['val'], label="Validation")
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            if ylim:
                ax.set_ylim(ylim)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(loc='upper left', frameon=True, fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.6)
        
        for ax, (title, (train_key, val_key)) in zip(axs.flat, metrics.items()):
            plot_metric(ax, history.history[train_key], history.history[val_key], title, title, ylim=(0, 1) if 'Dice' in title or 'IoU' in title else None)
        
        plt.savefig('training_metrics.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Learning Rate Plot
        plt.figure(figsize=(8, 4))
        plt.semilogy(epochs, history.history['lr'], marker='o', markersize=6, linewidth=2, color=colors["val"])
        plt.title("Learning Rate Schedule", fontsize=14, fontweight='bold')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Learning Rate", fontsize=12)
        plt.grid(True, which="both", linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('lr_metrics.png', format='png', dpi=300, bbox_inches='tight')
        plt.show()
