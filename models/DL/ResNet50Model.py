import os
import numpy as np
import tensorflow as tf

from keras.layers import Lambda, Input, Conv2D, UpSampling2D, Resizing, BatchNormalization, Activation
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import LearningRateScheduler, EarlyStopping

from sklearn.model_selection import train_test_split

from utils.SaveHistoryToTxt import save_history_to_txt
from utils.BinarySegmentationMetrics import BinarySegmentationMetrics
from utils.CreateDirectory import _create_directory
from models.DL.DeepLearningUtils.ImagePreprocessing import pair_images_by_filename
from keras.preprocessing.image import ImageDataGenerator


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def create_augmenter():
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )



class ResNet50Model:
    def __init__(self, rgb_dirs: list[str], disease_segmented_dirs: list[str], leaf_segmented_dirs: list[str], learning_rate: float, val_split: float, dataset_name: str) -> None:
        self.rgb_dirs = rgb_dirs
        self.disease_segmented_dirs = disease_segmented_dirs
        self.leaf_segmented_dirs = leaf_segmented_dirs
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.dataset_name = dataset_name
        self.model = self._build_model()

    def load_images_and_masks(self, paired_image_paths, target_size=(256, 256)):
        combined_images = []
        disease_masks = []
        disease_types = []

        for rgb_path, leaf_path, disease_path, disease_type in paired_image_paths:
            if all(os.path.exists(p) and p.endswith('.png') for p in [rgb_path, leaf_path, disease_path]):
                # Load RGB image and preprocess
                rgb_image = img_to_array(load_img(rgb_path, target_size=target_size, color_mode='rgb')) / 255.0
                
                # Load leaf mask, preprocess, and expand dimensions if necessary
                leaf_mask = img_to_array(load_img(leaf_path, target_size=target_size, color_mode='grayscale')) / 255.0
                if leaf_mask.ndim == 2:
                    leaf_mask = np.expand_dims(leaf_mask, axis=-1)

                # Combine RGB image and leaf mask along the channel axis
                combined_image = np.concatenate([rgb_image, leaf_mask], axis=-1)


                combined_images.append(combined_image)
                
                # Load and preprocess disease mask
                disease_mask = img_to_array(load_img(disease_path, target_size=target_size, color_mode='grayscale'))
                disease_mask = np.where(disease_mask > 127, 1, 0)
                if disease_mask.ndim == 2:  # Check if disease_mask is 2D
                    disease_mask = np.expand_dims(disease_mask, axis=-1)  # Add a single channel dimension to make it 3D

                # Append to lists
                disease_masks.append(disease_mask)
                disease_types.append(disease_type)
            else:
                print(f"Image not found: {rgb_path}, {leaf_path}, {disease_path}")

        # Normalize combined images for ResNet50
        combined_images = np.array(combined_images)  # No need to scale to [0, 255] range
        combined_images = preprocess_input(combined_images)  # Use ResNet50 preprocessing

        return np.array(combined_images), np.array(disease_masks), np.array(disease_types)


    def _build_model(self) -> Model:
        # Load pre-trained ResNet50 model without top layers
        base_model = ResNet50(weights='imagenet', include_top=False)

        # Freeze pre-trained layers
        for layer in base_model.layers:
            layer.trainable = False

        # Create new input layer for 6-channel input
        input_tensor = Input(shape=(256, 256, 6))

        # Use a Lambda layer to take only the first 3 channels (RGB) to feed into the ResNet50
        x = Lambda(lambda x: x[:, :, :, :3])(input_tensor)

        # Feed the RGB channels into the base model
        x = base_model(x)

        # Add custom layers to handle the output from the base model
        x = Conv2D(512, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D(size=(2, 2))(x)

        x = Conv2D(256, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D(size=(2, 2))(x)

        x = Resizing(256, 256)(x)
        disease_segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='disease_segmentation')(x)

        
        model = Model(inputs=input_tensor, outputs=disease_segmentation)

        return model

    def compile_and_train(self, epochs: int, batch_size: int, output_dir: str):
        # Directory setup
        _create_directory(output_dir)
        plots_dir = os.path.join(output_dir, 'plots')
        _create_directory(plots_dir)

        # Define learning rate schedule function
        def lr_schedule(epoch):
            if epoch < 10:
                return 0.001
            elif epoch < 20:
                return 0.0005
            else:
                return 0.0001

        # Create LearningRateScheduler callback
        lr_scheduler = LearningRateScheduler(lr_schedule)

        # Create EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_mean_io_u', patience=10)

        # Load and preprocess data
        all_paired_image_paths = pair_images_by_filename(self.rgb_dirs, self.disease_segmented_dirs, self.leaf_segmented_dirs)

        # Initialize empty lists for training and validation datasets
        disease_groups = {}
        for path in all_paired_image_paths:
            disease = path[3] 
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
        combined_inputs_train, disease_labels_train, train_disease_types = self.load_images_and_masks(
            stratified_train_data, target_size=(256, 256))
        combined_inputs_val, disease_labels_val, val_disease_types = self.load_images_and_masks(
            stratified_val_data, target_size=(256, 256))

        binary_segmentation_metrics = BinarySegmentationMetrics(validation_data=(combined_inputs_val, disease_labels_val), validation_disease_types=val_disease_types, model_name='ResNet50', learning_rate=self.learning_rate, val_split=self.val_split, dataset_name=self.dataset_name, output_dir=output_dir)
        
        augmenter = create_augmenter()

        def train_generator(data_generator, images, masks):
            seed = 1  # Ensuring that image and mask undergo the same transformation
            image_gen = data_generator.flow(images, batch_size=batch_size, seed=seed)
            mask_gen = data_generator.flow(masks, batch_size=batch_size, seed=seed)
            
            while True:
                yield next(image_gen), next(mask_gen)

        train_gen = train_generator(augmenter, combined_inputs_train, disease_labels_train)

        # Model compilation
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), 
                        loss=BinaryCrossentropy(),
                        metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])

        # Model training
        history = self.model.fit(
            train_gen,
            validation_data=(combined_inputs_val, disease_labels_val),
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=[binary_segmentation_metrics, lr_scheduler, early_stopping]  # Add lr_scheduler and early_stopping to callbacks
            )
        
        binary_segmentation_metrics.save_results_to_excel()

        # Save the model and training history
        save_history_to_txt(history, output_dir)
        
        tf.keras.backend.clear_session()

        return history

