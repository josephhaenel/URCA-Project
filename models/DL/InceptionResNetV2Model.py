import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, Conv2D, UpSampling2D, Resizing, concatenate, Multiply, GlobalAveragePooling2D, Dense, Reshape, Activation
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall
from tensorflow.keras.losses import BinaryCrossentropy
from utils.BinarySegmentationMetrics import BinarySegmentationMetrics
from kerastuner import RandomSearch, HyperParameters
from sklearn.model_selection import train_test_split
from utils.F1Score import F1Score
from utils.IoUMetric import IoUMetric, IoULogger
from utils.GraphPlotter import save_history_to_txt
from utils.BinarySegmentationMetrics import BinarySegmentationMetrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

# Suppress TensorFlow warnings for a cleaner output
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def dice_loss(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)  # Add this line
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true, axis=[1,2,3]) + tf.reduce_sum(y_pred, axis=[1,2,3])
    dice = tf.reduce_mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return 1 - dice

class InceptionResNetV2Model: 

    def __init__(self, rgb_dirs: list[str], disease_segmented_dirs: list[str], leaf_segmented_dirs: list[str], learning_rate: float, val_split: float, dataset_name: str) -> None:
        self.rgb_dirs = rgb_dirs
        self.disease_segmented_dirs = disease_segmented_dirs
        self.leaf_segmented_dirs = leaf_segmented_dirs
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.dataset_name = dataset_name
        self.model = self._build_model()
        
    def load_images_and_masks(self, paired_image_paths, target_size=(299, 299)) -> tuple[np.ndarray, np.ndarray, list]:
        combined_images = []
        disease_masks = []
        disease_types = []

        for rgb_path, leaf_mask_path, disease_path, disease_type in paired_image_paths:
            if os.path.exists(rgb_path) and os.path.exists(leaf_mask_path) and os.path.exists(disease_path):
                rgb_image = preprocess_input(img_to_array(load_img(rgb_path, target_size=target_size, color_mode='rgb')))
                
                leaf_mask = img_to_array(load_img(leaf_mask_path, target_size=target_size, color_mode='grayscale'))
                if leaf_mask.ndim == 2:  # Check if leaf_mask is 2D
                    leaf_mask = np.expand_dims(leaf_mask, axis=-1)  # Add a single channel dimension to make it 3D
                
                # Check the shape of leaf_mask to ensure it is 3D (height, width, channels)
                if leaf_mask.shape[-1] != 1:
                    raise ValueError(f"Leaf mask should have one channel. Found shape: {leaf_mask.shape}")
                
                combined_image = np.concatenate([rgb_image, leaf_mask], axis=-1)
                combined_images.append(combined_image)
                
                disease_mask = img_to_array(load_img(disease_path, target_size=(256, 256), color_mode='grayscale'))
                
                disease_mask = np.where(disease_mask > 127, 1, 0)
                disease_mask = np.expand_dims(disease_mask, axis=-1)  # Ensure disease mask has a single channel
                disease_masks.append(disease_mask)
                
                disease_types.append(disease_type)

        return np.array(combined_images), np.array(disease_masks), np.array(disease_types)

    def pair_images_by_filename(self, base_rgb_dir: str, base_disease_dir: str, base_leaf_dir: str) -> list[tuple[str, str, str, str]]:
        paired_images = []
        for disease in os.listdir(base_rgb_dir):
            rgb_dir = os.path.join(base_rgb_dir, disease)
            disease_dir = os.path.join(base_disease_dir, disease)
            leaf_dir = os.path.join(base_leaf_dir, disease)

            # Ensure directories for RGB, disease, and leaf images exist
            if not os.path.isdir(rgb_dir) or not os.path.isdir(disease_dir) or not os.path.isdir(leaf_dir):
                print(f"One of the directories is invalid: {rgb_dir}, {disease_dir}, {leaf_dir}")
                continue

            # Iterate over RGB images and match with corresponding disease and leaf images
            for file_name in os.listdir(rgb_dir):
                if file_name.endswith('.png'):
                    rgb_path = os.path.join(rgb_dir, file_name)
                    disease_path = os.path.join(disease_dir, file_name)
                    leaf_path = os.path.join(leaf_dir, file_name)

                    # Ensure paths for RGB, disease, and leaf images exist before adding
                    if os.path.exists(rgb_path) and os.path.exists(disease_path) and os.path.exists(leaf_path):
                        paired_images.append((rgb_path, leaf_path, disease_path, disease))
                    else:
                        print(f"Missing image for {file_name} in {disease}")

        return paired_images


    def load_images(self, image_paths: list[str], is_mask: bool = False, target_size: tuple[int, int] = (299, 299)) -> np.ndarray:
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

    def _build_model(self):
        # Input tensor for RGB images and leaf segmentation mask
        input_tensor = Input(shape=(299, 299, 4))

        # Split RGB and mask
        processed_rgb = Lambda(lambda x: x[..., :3])(input_tensor)
        processed_mask = Lambda(lambda x: x[..., 3:])(input_tensor)

        # Process RGB channels with InceptionResNetV2
        base_model_rgb = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=processed_rgb)
        rgb_features = base_model_rgb.output

        # Process the leaf segmentation mask through separate layers
        mask_conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(processed_mask)
        mask_conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(mask_conv1)
        
        mask_conv_resized = Resizing(8, 8)(mask_conv2)

        # Combine base_model_rgb output with processed mask features
        combined_features = concatenate([rgb_features, mask_conv_resized])

        # Attention Mechanism
        attention_map = Conv2D(1, (1, 1), activation='sigmoid')(combined_features)
        attention_features = Multiply()([combined_features, attention_map])

        x = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(attention_features)
        x = tf.keras.layers.BatchNormalization()(x)  # Add Batch Normalization
        x = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
        x = tf.keras.layers.Dropout(0.5)(x)  # Add another dropout layer
        x = UpSampling2D(size=(2, 2))(x)
        x = Resizing(256, 256)(x)

        # Final output layer for disease segmentation
        disease_segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='disease_segmentation')(x)

        return Model(inputs=input_tensor, outputs=disease_segmentation)

    def _create_directory(self, path: str) -> None:
        """
        Ensures a directory exists. Creates one if it does not exist.

        Parameters:
            path (str): The path of the directory to be checked or created.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def print_history_keys(self, history: tf.keras.callbacks.History) -> None:
        """
        Prints keys from the training history. For debugging purposes.

        Parameters:
            history (tf.keras.callbacks.History): The history object from model training.
        """
        print("Keys in training history:")
        for key in history.history.keys():
            print(key)

    def compile_and_train(self, epochs: int, batch_size: int, output_dir: str):
        # Directory setup
        self._create_directory(output_dir)
        plots_dir = os.path.join(output_dir, 'plots')
        self._create_directory(plots_dir)

        # Create EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_mean_io_u', patience=25)

        # Load and preprocess data
        all_paired_image_paths = self.pair_images_by_filename(self.rgb_dirs, self.disease_segmented_dirs, self.leaf_segmented_dirs)

        # Initialize empty lists for training and validation datasets
        disease_groups = {}
        for path in all_paired_image_paths:
            disease = path[3]  # Assuming the disease type is indeed the fourth element
            if disease not in disease_groups:
                disease_groups[disease] = []
            disease_groups[disease].append(path)

        # Initialize empty lists for stratified training and validation datasets
        stratified_train_data = []
        stratified_val_data = []

        # Split each group into stratified training and validation sets
        for disease, paths in disease_groups.items():
            train_paths, val_paths = train_test_split(
                paths, 
                test_size=self.val_split, 
                stratify=[p[3] for p in paths],  # Stratify by the disease type
                random_state=42
            )
            stratified_train_data.extend(train_paths)
            stratified_val_data.extend(val_paths)

        # Preparing training and validation datasets
        combined_inputs_train, disease_labels_train, train_disease_types = self.load_images_and_masks(stratified_train_data)
        combined_inputs_val, disease_labels_val, val_disease_types = self.load_images_and_masks(stratified_val_data)

        binary_segmentation_metrics = BinarySegmentationMetrics(validation_data=(combined_inputs_val, disease_labels_val), validation_disease_types=val_disease_types, model_name = 'InceptionResNetV2', learning_rate=self.learning_rate, val_split=self.val_split, dataset_name=self.dataset_name, output_dir=output_dir)
        
        # Model compilation
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
                        loss=dice_loss,
                        metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])

        # Fits the model on batches with real-time data augmentation
        history = self.model.fit(
                        combined_inputs_train, 
            disease_labels_train, 
            validation_data=(combined_inputs_val, disease_labels_val),
            epochs=epochs, 
            callbacks=[binary_segmentation_metrics, early_stopping]  # Add early_stopping to callbacks
            )

        # Unfreeze all layers
        for layer in self.model.layers:
            layer.trainable = True

        # Recompile the model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
                        loss=dice_loss,
                        metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])

        # Fit the model again
        history = self.model.fit(
                        combined_inputs_train, 
            disease_labels_train, 
            validation_data=(combined_inputs_val, disease_labels_val),
            epochs=epochs, 
            callbacks=[binary_segmentation_metrics, early_stopping]  # Add early_stopping to callbacks
            )
        
        binary_segmentation_metrics.save_results_to_excel()

        # Save the model and training history
        save_history_to_txt(history, output_dir)

        return history




