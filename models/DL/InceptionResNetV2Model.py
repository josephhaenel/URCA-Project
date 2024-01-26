import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, Conv2D, UpSampling2D, Resizing
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from utils.F1Score import F1Score
from utils.GraphPlotter import save_plots, save_history_to_txt
from tensorflow.keras.metrics import Recall
from tensorflow.keras.losses import BinaryCrossentropy

# Suppressing TensorFlow warnings for a cleaner output
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class CalculateIOU:
    @staticmethod
    def calculate_iou(y_true, y_pred):
        y_true_float = tf.cast(y_true, tf.float32)
        y_pred_float = tf.cast(y_pred, tf.float32)

        y_pred_thresholded = tf.cast(tf.greater(y_pred_float, 0.5), tf.float32)

        intersection = tf.reduce_sum(y_true_float * y_pred_thresholded)
        union = tf.reduce_sum(y_true_float) + tf.reduce_sum(y_pred_thresholded) - intersection

        iou_score = intersection / union
        return iou_score



class InceptionResNetV2Model:
    """
    A class for creating and training an InceptionResNetV2-based model for image segmentation.

    Attributes:
        rgb_dirs (list): Directory paths for RGB images.
        disease_segmented_dirs (list): Directory paths for disease-segmented images.
        leaf_segmented_dirs (list): Directory paths for leaf-segmented images.
        model (Model): The TensorFlow Keras model.

    Methods:
        load_images_and_masks: Loads and processes images and masks.
        pair_images_by_filename: Pairs images by filename from given directories.
        load_images: Loads and preprocesses images.
        _build_model: Constructs the InceptionResNetV2-based model.
        _create_directory: Ensures a directory exists.
        print_history_keys: Prints keys from the training history.
        compile_and_train: Compiles and trains the model.
    """

    def __init__(self, rgb_dirs: list, disease_segmented_dirs: list, leaf_segmented_dirs: list) -> None:
        self.rgb_dirs = rgb_dirs
        self.disease_segmented_dirs = disease_segmented_dirs
        self.leaf_segmented_dirs = leaf_segmented_dirs
        self.model = self._build_model()
        
    def load_images_and_masks(self, paired_image_paths, target_size=(299, 299)) -> np.ndarray:
        """
        Loads and processes images and masks from given file paths.

        Parameters:
        - paired_image_paths (list of tuples): Each tuple contains paths for the RGB image, leaf mask, and disease mask.
        - target_size (tuple): The target size for resizing the images.

        Returns:
        - np.ndarray: Array of combined RGB and mask images.
        """
        combined_images = []
        for rgb_path, leaf_path, disease_path in paired_image_paths:
            # Check if all paths exist
            if os.path.exists(rgb_path) and os.path.exists(leaf_path) and os.path.exists(disease_path):
                # Load and preprocess images
                rgb_image = preprocess_input(img_to_array(load_img(rgb_path, target_size=target_size, color_mode='rgb')))
                leaf_mask = np.repeat(img_to_array(load_img(leaf_path, target_size=target_size, color_mode='grayscale')) / 255.0, 3, axis=-1)
                disease_mask = np.repeat(img_to_array(load_img(disease_path, target_size=target_size, color_mode='grayscale')) / 255.0, 3, axis=-1)

                # Concatenate images
                combined_image = np.concatenate([rgb_image, leaf_mask, disease_mask], axis=-1)
                combined_images.append(combined_image)
            else:
                print(f"Image not found: {rgb_path}, {leaf_path}, {disease_path}")
        return np.array(combined_images)


    def pair_images_by_filename(self, base_rgb_dir, base_disease_dir, base_leaf_dir):
        paired_images = []
        for disease in os.listdir(base_rgb_dir):
            rgb_dir = os.path.join(base_rgb_dir, disease)
            disease_dir = os.path.join(base_disease_dir, disease)
            leaf_dir = os.path.join(base_leaf_dir, disease)

            if not os.path.isdir(rgb_dir) or not os.path.isdir(disease_dir) or not os.path.isdir(leaf_dir):
                print(f"One of the directories is invalid: {rgb_dir}, {disease_dir}, {leaf_dir}")
                continue

            rgb_files = {f for f in os.listdir(rgb_dir) if f.endswith('.png')}
            disease_files = {f for f in os.listdir(disease_dir) if f.endswith('.png')}
            leaf_files = {f for f in os.listdir(leaf_dir) if f.endswith('.png')}

            common_files = rgb_files.intersection(disease_files, leaf_files)
            paired_images.extend([(os.path.join(rgb_dir, f), os.path.join(disease_dir, f), os.path.join(leaf_dir, f)) for f in common_files])

        return paired_images

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
        Builds and returns the modified InceptionResNetV2 model.

        Returns:
        - Model: The constructed TensorFlow Keras model.
        """
        
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        
        input_tensor = Input(shape=(299, 299, 9))

        # Custom layer to process the 6-channel input
        processed_input = Lambda(lambda x: x[:, :, :, :3])(input_tensor)

        # Load the base InceptionResNetV2 model with the processed input
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=processed_input)

        x = base_model.output
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Resizing(256, 256)(x)

        disease_segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='disease_segmentation')(x)
        model = Model(inputs=base_model.input, outputs=disease_segmentation)
        return model

    def _create_directory(self, path):
        """
        Create a directory if it does not exist.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def print_history_keys(self, history):
        print("Keys in training history:")
        for key in history.history.keys():
            print(key)

    def compile_and_train(self, epochs: int, batch_size: int, output_dir: str) -> tf.keras.callbacks.History:
        # Directory setup
        self._create_directory(output_dir)
        plots_dir = os.path.join(output_dir, 'plots')
        self._create_directory(plots_dir)

        # Load and preprocess data
        paired_image_paths = self.pair_images_by_filename(self.rgb_dirs, self.disease_segmented_dirs, self.leaf_segmented_dirs)
        combined_inputs = self.load_images_and_masks(paired_image_paths)
        disease_labels = self.load_images([d[2] for d in paired_image_paths], is_mask=True, target_size=(256, 256))

        # Model compilation
        disease_metrics = [BinaryCrossentropy(), 'accuracy', F1Score(), Recall(name='recall')]
        self.model.compile(optimizer='adam', 
                        loss=BinaryCrossentropy(),
                        metrics=disease_metrics)

        # Model training
        history = self.model.fit(
            combined_inputs, 
            disease_labels, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.3)

        # Saving training metrics plots
        save_plots(history, plots_dir)
        save_history_to_txt(history, output_dir)
        predictions = self.model.predict(combined_inputs)
        iou_score = CalculateIOU.calculate_iou(disease_labels, predictions)
        print(f"Validation IoU for disease segmentation: {iou_score.numpy()}")

        self.print_history_keys(history)

        return history


