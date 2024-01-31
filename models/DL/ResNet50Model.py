import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, Conv2D, UpSampling2D, Resizing, BatchNormalization, Activation
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from utils.F1Score import F1Score
from utils.IoUMetric import IoUMetric, IoULogger
from utils.GraphPlotter import save_history_to_txt
from tensorflow.keras.metrics import Recall

# Suppress TensorFlow warnings for a cleaner output
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class ResNet50Model:
    """
    A class for creating and training a ResNet50-based model for image segmentation.

    Attributes:
        rgb_dirs (list[str]): Directory paths for RGB images.
        disease_segmented_dirs (list[str]): Directory paths for disease-segmented images.
        leaf_segmented_dirs (list[str]): Directory paths for leaf-segmented images.
        learning_rate (float): Learning rate for the model training.
        val_split (float): Validation split for the model training.
        model (Model): The TensorFlow Keras model.

    Methods:
        load_images_and_masks: Loads and processes images and masks.
        pair_images_by_filename: Pairs images by filename from given directories.
        load_images: Loads and preprocesses images.
        _build_model: Constructs the ResNet50-based model.
        _create_directory: Ensures a directory exists.
        compile_and_train: Compiles and trains the model.
    """

    def __init__(self, rgb_dirs: list[str], disease_segmented_dirs: list[str], leaf_segmented_dirs: list[str], 
                 learning_rate: float, val_split: float) -> None:
        self.rgb_dirs = rgb_dirs
        self.disease_segmented_dirs = disease_segmented_dirs
        self.leaf_segmented_dirs = leaf_segmented_dirs
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.model = self._build_model()
        

    def load_images_and_masks(self, paired_image_paths: list[tuple[str, str, str]], target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Loads and processes images and masks from given file paths.

        Parameters:
            paired_image_paths (list[tuple[str, str, str]]): Paths for RGB image, leaf mask, and disease mask.
            target_size (tuple[int, int]): Target size for resizing the images.

        Returns:
            np.ndarray: Array of preprocessed images.
        """
        preprocessed_images = []
        for rgb_path, leaf_path, disease_path in paired_image_paths:
            if all(os.path.exists(p) and p.endswith('.png') for p in [rgb_path, leaf_path, disease_path]):
                # Load and preprocess images
                rgb_image = load_img(rgb_path, target_size=target_size, color_mode='rgb')
                leaf_mask = load_img(leaf_path, target_size=target_size, color_mode='grayscale')
                disease_mask = load_img(disease_path, target_size=target_size, color_mode='grayscale')

                # Convert to array and normalize
                rgb_image = img_to_array(rgb_image) / 255.0
                leaf_mask = img_to_array(leaf_mask) / 255.0
                disease_mask = img_to_array(disease_mask) / 255.0

                rgb_image = rgb_image * leaf_mask * disease_mask

                # Preprocess for ResNet50
                preprocessed_image = preprocess_input(rgb_image * 255)  # Rescale to original range

                preprocessed_images.append(preprocessed_image)
            else:
                print(f"Image not found: {rgb_path}, {leaf_path}, {disease_path}")
        return np.array(preprocessed_images)


    def pair_images_by_filename(self, base_rgb_dir: str, base_disease_dir: str, base_leaf_dir: str) -> list[tuple[str, str, str]]:
        """
        Pairs images by filename from given directories.

        Parameters:
            base_rgb_dir (str): Base directory path for RGB images.
            base_disease_dir (str): Base directory path for disease-segmented images.
            base_leaf_dir (str): Base directory path for leaf-segmented images.

        Returns:
            list[tuple[str, str, str]]: List of tuples containing paths of paired RGB, disease, and leaf images.
        """
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
    
    def load_images(self, image_paths: list[str], is_mask: bool = False, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Loads and preprocesses images.

        Parameters:
            image_paths (list[str]): List of image file paths.
            is_mask (bool): Specifies whether the images are masks.
            target_size (tuple[int, int]): Target size for resizing the images.

        Returns:
            np.ndarray: Numpy array of loaded and preprocessed images.
        """
        images = []
        for image_path in image_paths:
            if os.path.exists(image_path) and image_path.endswith('.png'):
                image = load_img(image_path, target_size=target_size, color_mode='grayscale' if is_mask else 'rgb')
                image = img_to_array(image)
                if is_mask:
                    # Resize mask to match model output
                    image = tf.image.resize(image, (224, 224))
                image = image / 255.0 if is_mask else preprocess_input(image)
                images.append(image)
            else:
                print(f"Image not found: {image_path}")
        return np.array(images)

    def _build_model(self) -> Model:
        """
        Constructs the ResNet50-based model.

        Returns:
            Model: The constructed TensorFlow Keras model.
        """
        input_tensor = Input(shape=(224, 224, 3))
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=input_tensor)

        x = base_model.output
        x = Conv2D(512, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D(size=(2, 2))(x)

        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D(size=(2, 2))(x)

        x = Resizing(224, 224)(x)
        disease_segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='disease_segmentation')(x)

        model = Model(inputs=base_model.input, outputs=disease_segmentation)
        return model

    def _create_directory(self, path: str) -> None:
        """
        Ensures a directory exists. Creates one if it does not exist.

        Parameters:
            path (str): The path of the directory to be checked or created.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def compile_and_train(self, epochs: int, batch_size: int, output_dir: str) -> tf.keras.callbacks.History:
        """
        Compiles and trains the model.

        Parameters:
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.
            output_dir (str): Output directory to save training artifacts.

        Returns:
            tf.keras.callbacks.History: History object containing training metrics.
        """
        # Directory setup
        self._create_directory(output_dir)

        # Load and preprocess data
        paired_image_paths = self.pair_images_by_filename(self.rgb_dirs, self.disease_segmented_dirs, self.leaf_segmented_dirs)
        combined_inputs = self.load_images_and_masks(paired_image_paths)
        disease_labels = self.load_images([d[2] for d in paired_image_paths], is_mask=True, target_size=(256, 256))
        leaf_labels = self.load_images([d[1] for d in paired_image_paths], is_mask=True, target_size=(256, 256))
        
        total_size = len(combined_inputs)
        training_size = int(total_size * (1 - self.val_split))
        validation_size = total_size - training_size
        
        print(f"Total dataset size: {total_size}")
        print(f"Training dataset size: {training_size}")
        print(f"Validation dataset size: {validation_size}")

        # Model compilation
        disease_metrics = ['accuracy', F1Score(), Recall(name='recall'), IoUMetric()]

        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=BinaryCrossentropy(),
            metrics=disease_metrics)
        
        iou_logger = IoULogger(output_dir)

        # Model training
        history = self.model.fit(
            combined_inputs, 
            disease_labels, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=self.val_split,
            callbacks=[iou_logger])

        # Save training metrics and history
        save_history_to_txt(history, output_dir)

        return history
