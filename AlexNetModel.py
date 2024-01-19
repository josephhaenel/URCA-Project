import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from F1Score import F1Score
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Resizing, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.callbacks import Callback

class AlexNetModel:
    def __init__(self, rgb_dir: str, disease_segmented_dir: str, leaf_segmented_dir: str) -> None:
        """
        Initialize the AlexNetModel with directories for RGB, disease-segmented,
        and leaf-segmented images.

        Parameters:
        - rgb_dir (str): Directory path for RGB images.
        - disease_segmented_dir (str): Directory path for disease-segmented images.
        - leaf_segmented_dir (str): Directory path for leaf-segmented images.
        """
        self.rgb_dir = rgb_dir
        self.disease_segmented_dir = disease_segmented_dir
        self.leaf_segmented_dir = leaf_segmented_dir
        self.model = self._build_model()

    def load_images(self, base_dir: str, subfolders: list, is_mask: bool = False, target_size: tuple = (227, 227)) -> np.ndarray:
        """
        Load and preprocess images from specified subdirectories.

        Parameters:
        - base_dir (str): Base directory containing image subfolders.
        - subfolders (list): List of subfolder names within the base directory.
        - is_mask (bool): Flag indicating if the images are masks. Defaults to False.
        - target_size (tuple): Target size for resizing the images. Defaults to (227, 227).

        Returns:
        - np.ndarray: Array of preprocessed image data.
        """
        images = []
        for subfolder in subfolders:
            image_dir = os.path.join(base_dir, subfolder)
            for filename in os.listdir(image_dir):
                if filename.endswith('.png'):
                    image_path = os.path.join(image_dir, filename)
                    image = load_img(image_path, target_size=target_size, color_mode='grayscale' if is_mask else 'rgb')
                    image = img_to_array(image)
                    image = image / 255.0 if is_mask else preprocess_input(image, mode='caffe')
                    images.append(image)
        return np.array(images)

    def _build_model(self) -> Model:
        """
        Constructs and returns the AlexNet-like model modified for image segmentation.

        Returns:
        - Model: The constructed TensorFlow Keras model.
        """
        input_tensor = Input(shape=(227, 227, 3))
        x = Conv2D(96, (11, 11), strides=4, activation='relu')(input_tensor)
        x = MaxPooling2D((3, 3), strides=2)(x)
        x = Conv2D(256, (5, 5), padding='same', activation='relu')(x)
        x = MaxPooling2D((3, 3), strides=2)(x)
        x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D((3, 3), strides=2)(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Resizing(256, 256)(x)
        disease_segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='disease_segmentation')(x)
        leaf_segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='leaf_segmentation')(x)

        model = Model(inputs=input_tensor, outputs=[disease_segmentation, leaf_segmentation])
        return model

    def _create_directory(self, path):
        """
        Create a directory if it does not exist.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def _save_plots(self, history, output_dir):
        """
        Save plots for training and validation metrics.
        """
        # Loss Plot
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.savefig(os.path.join(output_dir, 'training_plots.png'))

    def _save_segmentation_images(self, images, predictions, output_dir, epoch):
        """
        Save generated segmentation images.
        """
        for i in range(len(images)):
            plt.imsave(os.path.join(output_dir, f'epoch_{epoch}_image_{i}.png'), predictions[i], cmap='gray')

    def compile_and_train(self, epochs: int, batch_size: int, output_dir: str):
        """
        Compiles and trains the model, and saves outputs to the specified directory.
        """
        # Directory setup
        self._create_directory(output_dir)
        plots_dir = os.path.join(output_dir, 'plots')
        images_dir = os.path.join(output_dir, 'images')
        self._create_directory(plots_dir)
        self._create_directory(images_dir)

        # Load and preprocess data
        subfolders = os.listdir(self.rgb_dir)
        images = self.load_images(self.rgb_dir, subfolders)
        disease_labels = self.load_images(self.disease_segmented_dir, subfolders, is_mask=True, target_size=(256, 256))
        leaf_segmentation_masks = self.load_images(self.leaf_segmented_dir, subfolders, is_mask=True, target_size=(256, 256))

        # Model compilation
        f1_score_metric = F1Score()
        self.model.compile(optimizer='adam',
                           loss={'disease_segmentation': 'binary_crossentropy',
                                 'leaf_segmentation': 'binary_crossentropy'},
                           metrics=['accuracy', f1_score_metric, tf.keras.metrics.Recall()])

        # Model training
        history = self.model.fit(images, 
                                 {'disease_segmentation': disease_labels, 'leaf_segmentation': leaf_segmentation_masks},
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_split=0.1)  # Assuming a validation split

        # Saving training metrics plots
        self._save_plots(history, plots_dir)

        # Generating and saving segmentation images (post-training)
        predictions = self.model.predict(images)
        self._save_segmentation_images(images, predictions, images_dir, epochs)

        return history