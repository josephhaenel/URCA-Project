import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, Conv2D, UpSampling2D, Resizing, BatchNormalization, Activation
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from utils.SaveHistoryToTxt import save_history_to_txt
from sklearn.model_selection import train_test_split
from utils.BinarySegmentationMetrics import BinarySegmentationMetrics
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

class ResNet50Model:
    def __init__(self, rgb_dirs: list[str], disease_segmented_dirs: list[str], leaf_segmented_dirs: list[str], learning_rate: float, val_split: float, dataset_name: str) -> None:
        self.rgb_dirs = rgb_dirs
        self.disease_segmented_dirs = disease_segmented_dirs
        self.leaf_segmented_dirs = leaf_segmented_dirs
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.dataset_name = dataset_name
        self.model = self._build_model()

    def load_images_and_masks(self, paired_image_paths, target_size=(224, 224)):
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

                # Check if RGB and leaf mask have the same number of channels
                if rgb_image.shape[-1] == leaf_mask.shape[-1]:
                    # Combine RGB image and leaf mask
                    combined_image = np.concatenate([rgb_image, leaf_mask], axis=-1)  # Combine along the channel axis
                else:
                    # Expand leaf mask to match RGB image's channel dimension
                    expanded_leaf_mask = np.tile(leaf_mask, (1, 1, rgb_image.shape[-1]))
                    # Combine RGB image and expanded leaf mask
                    combined_image = np.concatenate([rgb_image, expanded_leaf_mask], axis=-1)

                # Load and preprocess disease mask
                disease_mask = img_to_array(load_img(disease_path, target_size=target_size, color_mode='grayscale'))
                disease_mask = np.where(disease_mask > 127, 1, 0)
                disease_mask = np.expand_dims(disease_mask, axis=-1)

                # Append to lists
                combined_images.append(combined_image)
                disease_masks.append(disease_mask)
                disease_types.append(disease_type)
            else:
                print(f"Image not found: {rgb_path}, {leaf_path}, {disease_path}")

        # Normalize combined images for ResNet50
        combined_images = np.array(combined_images)  # No need to scale to [0, 255] range
        combined_images = preprocess_input(combined_images)  # Use ResNet50 preprocessing

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

    
    def load_images(self, image_paths: list[str], is_mask: bool = False, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
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
        # Load pre-trained ResNet50 model without top layers
        base_model = ResNet50(weights='imagenet', include_top=False)

        # Freeze pre-trained layers
        for layer in base_model.layers:
            layer.trainable = False

        # Create new input layer for 6-channel input
        input_tensor = Input(shape=(224, 224, 6))

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

        x = Resizing(224, 224)(x)
        disease_segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='disease_segmentation')(x)

        
        model = Model(inputs=input_tensor, outputs=disease_segmentation)

        return model


    def _create_directory(self, path: str) -> None:
        """
        Ensures a directory exists. Creates one if it does not exist.

        Parameters:
            path (str): The path of the directory to be checked or created.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def compile_and_train(self, epochs: int, batch_size: int, output_dir: str):
        # Directory setup
        self._create_directory(output_dir)
        plots_dir = os.path.join(output_dir, 'plots')
        self._create_directory(plots_dir)

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
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)

        # Load and preprocess data
        all_paired_image_paths = self.pair_images_by_filename(self.rgb_dirs, self.disease_segmented_dirs, self.leaf_segmented_dirs)

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
        combined_inputs_train, disease_labels_train, train_disease_types = self.load_images_and_masks(stratified_train_data)
        combined_inputs_val, disease_labels_val, val_disease_types = self.load_images_and_masks(stratified_val_data)

        binary_segmentation_metrics = BinarySegmentationMetrics(validation_data=(combined_inputs_val, disease_labels_val), validation_disease_types=val_disease_types, model_name='ResNet50', learning_rate=self.learning_rate, val_split=self.val_split, dataset_name=self.dataset_name, output_dir=output_dir)
        
        # Model compilation
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), 
                        loss=BinaryCrossentropy(),
                        metrics=['accuracy'])

        # Model training
        history = self.model.fit(
            combined_inputs_train, 
            disease_labels_train, 
            validation_data=(combined_inputs_val, disease_labels_val),
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=[binary_segmentation_metrics, lr_scheduler, early_stopping]  # Add lr_scheduler and early_stopping to callbacks
            )
        
        binary_segmentation_metrics.save_results_to_excel()

        # Save the model and training history
        save_history_to_txt(history, output_dir)

        return history

