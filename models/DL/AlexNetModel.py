import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, Conv2D, MaxPooling2D, UpSampling2D, Resizing, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from utils.F1Score import F1Score
from utils.GraphPlotter import save_history_to_txt
from tensorflow.keras.metrics import Recall
from tensorflow.keras.losses import BinaryCrossentropy
from utils.IoUMetric import IoUMetric
from tensorflow.keras.optimizers import Adam

class CalculateIOU:
    @staticmethod
    def calculate_iou(y_true, y_pred):
        y_true_float = tf.cast(y_true, tf.float32)
        y_pred_float = tf.cast(y_pred, tf.float32)

        y_pred_thresholded = tf.cast(tf.greater(y_pred_float, 0.5), tf.float32)

        intersection = tf.reduce_sum(y_true_float * y_pred_thresholded)
        union = tf.reduce_sum(y_true_float) + tf.reduce_sum(y_pred_thresholded) - intersection

        # Compute IoU
        iou_score = intersection / union
        return iou_score



class AlexNetModel:
    """
    A class to create and train an AlexNet-based model for image segmentation.

    Attributes:
        rgb_dirs (str): Directory path for RGB images.
        disease_segmented_dirs (str): Directory path for disease segmented images.
        leaf_segmented_dirs (str): Directory path for leaf segmented images.
        model (tf.keras.Model): The AlexNet-based segmentation model.

    Methods:
        pair_images_by_filename: Pairs images by filenames from given directories.
        load_images_and_masks: Loads images and masks for training.
        load_images: Loads images from given paths.
        _build_model: Builds the AlexNet-based segmentation model.
        _create_directory: Creates a directory if it does not exist.
        compile_and_train: Compiles and trains the model.
    """

    def __init__(self, rgb_dirs, disease_segmented_dirs, leaf_segmented_dirs):
        self.rgb_dirs = rgb_dirs
        self.disease_segmented_dirs = disease_segmented_dirs
        self.leaf_segmented_dirs = leaf_segmented_dirs
        self.model = self._build_model()
        
    def pair_images_by_filename(self, base_rgb_dir, base_disease_dir, base_leaf_dir):
        paired_images = []
        for disease_folder in os.listdir(base_rgb_dir):
            rgb_dir = os.path.join(base_rgb_dir, disease_folder)
            disease_dir = os.path.join(base_disease_dir, disease_folder)
            leaf_dir = os.path.join(base_leaf_dir, disease_folder)

            # Check if directories are valid
            if not os.path.isdir(rgb_dir) or not os.path.isdir(disease_dir) or not os.path.isdir(leaf_dir):
                print(f"One of the directories is invalid: {rgb_dir}, {disease_dir}, {leaf_dir}")
                continue

            rgb_files = {f for f in os.listdir(rgb_dir) if f.endswith('.png')}
            disease_files = {f for f in os.listdir(disease_dir) if f.endswith('.png')}
            leaf_files = {f for f in os.listdir(leaf_dir) if f.endswith('.png')}

            common_files = rgb_files.intersection(disease_files, leaf_files)
            print(f"Found {len(common_files)} common files in folder {disease_folder}")

            for f in common_files:
                paired_images.append((os.path.join(rgb_dir, f), os.path.join(disease_dir, f), os.path.join(leaf_dir, f)))
                print(f"Paired: {os.path.join(rgb_dir, f)}, {os.path.join(disease_dir, f)}, {os.path.join(leaf_dir, f)}")

        return paired_images


    def load_images_and_masks(self, paired_image_paths, target_size=(227, 227)):
        combined_images = []
        disease_masks = []
        for rgb_path, disease_path, leaf_path in paired_image_paths:
            if os.path.exists(rgb_path) and os.path.exists(leaf_path) and os.path.exists(disease_path):
                rgb_image = load_img(rgb_path, target_size=target_size, color_mode='rgb')
                leaf_mask = load_img(leaf_path, target_size=target_size, color_mode='grayscale')
                disease_mask = load_img(disease_path, target_size=target_size, color_mode='grayscale')

                rgb_image = img_to_array(rgb_image)
                leaf_mask = img_to_array(leaf_mask)
                disease_mask = img_to_array(disease_mask)

                rgb_image = preprocess_input(rgb_image)
                leaf_mask = np.repeat(leaf_mask / 255.0, 3, axis=-1)

                combined_image = np.concatenate([rgb_image, leaf_mask], axis=-1)
                combined_images.append(combined_image)
                disease_masks.append(disease_mask / 255.0)
            else:
                print(f"Image not found: {rgb_path}, {disease_path}, {leaf_path}")
                print(f"Total loaded images: {len(combined_images)}")
            print(f"Total loaded masks: {len(disease_masks)}")
        return np.array(combined_images), np.array(disease_masks)
    
    def load_images(self, image_paths, is_mask: bool = False, target_size: tuple = (299, 299)) -> np.ndarray:
        images = []
        for image_path in image_paths:
            if os.path.exists(image_path) and image_path.endswith('.png'):
                image = load_img(image_path, target_size=target_size, color_mode='grayscale' if is_mask else 'rgb')
                image = img_to_array(image)
                image = image / 255.0 if is_mask else preprocess_input(image)
                images.append(image)
            else:
                print(f"Image not found: {image_path}")
        return np.array(images)

    def _build_model(self) -> Model:
        """
        Builds the AlexNet-based segmentation model.

        Returns:
            Model: A tf.keras.Model object representing the segmentation model.
        """
        input_tensor = Input(shape=(227, 227, 6))  # Adjust input size for AlexNet dimensions

        # Extract RGB and mask channels from the input
        processed_rgb = Lambda(lambda x: x[:, :, :, :3])(input_tensor)
        processed_mask = Lambda(lambda x: x[:, :, :, 3:])(input_tensor)

        # AlexNet-like architecture
        x = Conv2D(96, (11, 11), strides=4, activation='relu')(processed_rgb)
        x = MaxPooling2D((3, 3), strides=2)(x)
        x = Conv2D(256, (5, 5), padding='same', activation='relu')(x)
        x = MaxPooling2D((3, 3), strides=2)(x)
        x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D((3, 3), strides=2)(x)

        # UpSampling for Segmentation
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Resizing(227, 227)(x)

        # Combine features with the mask
        combined = concatenate([x, processed_mask])

        # Output layer for segmentation
        disease_segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='disease_segmentation')(combined)

        return Model(inputs=input_tensor, outputs=disease_segmentation)

    def _create_directory(self, path):
        """
        Create a directory if it does not exist.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def compile_and_train(self, epochs, batch_size, output_dir):
        # Directory setup
        self._create_directory(output_dir)
        plots_dir = os.path.join(output_dir, 'plots')
        self._create_directory(plots_dir)

        # Generate paired image paths
        paired_image_paths = self.pair_images_by_filename(self.rgb_dirs, self.disease_segmented_dirs, self.leaf_segmented_dirs)

        # Load and preprocess data
        combined_inputs, disease_labels = self.load_images_and_masks(paired_image_paths, target_size=(227, 227))

        # Model compilation
        disease_metrics = [BinaryCrossentropy(), 'accuracy', F1Score(), Recall(name='recall'), IoUMetric()]
        self.model.compile(optimizer=Adam(learning_rate=0.0000005),
                        loss={'disease_segmentation': BinaryCrossentropy()},
                        metrics={'disease_segmentation': disease_metrics}
                        )

        # Model training
        history = self.model.fit(combined_inputs, 
                                 disease_labels,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 validation_split=0.5)

        # Save training metrics and history
        save_history_to_txt(history, output_dir)
        predictions = self.model.predict(combined_inputs)
        iou_score = CalculateIOU.calculate_iou(disease_labels, predictions[0])
        print(f"Validation IoU for disease segmentation: {iou_score.numpy()}")

        return history