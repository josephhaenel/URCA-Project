import os
import numpy as np
import tensorflow as tf

from keras.layers import Lambda, Input, Conv2D, UpSampling2D, Resizing, BatchNormalization, Activation
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import LearningRateScheduler, EarlyStopping

from sklearn.model_selection import train_test_split

from utils.SaveHistoryToTxt import save_history_to_txt
from utils.BinarySegmentationMetrics import BinarySegmentationMetrics
from utils.CreateDirectory import _create_directory
from models.DL.DeepLearningUtils.ImagePreprocessing import pair_images_by_filename
from keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import pandas as pd
import cv2
from models.DL.DeepLearningUtils.CyclicLRChanger import CyclicLR




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
    
def iou_loss(y_true, y_pred):
    """
    Intersection-over-union (IoU) loss.

    Calculates the IoU score as 1 - iou_score.
    IoU is also known as Jaccard index and is a measure of overlap between two
    binary masks. It is calculated as:
    
        IoU = |A ∩ B| / |A ∪ B|

    where A and B are the ground truth mask and the predicted mask, respectively.

    Args:
        y_true: The ground truth tensor.
        y_pred: The predicted tensor.

    Returns:
        The scalar IoU loss.
    """
    def f(y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        return 1. - intersection / (union + tf.keras.backend.epsilon())

    return tf.reduce_mean(f(y_true, y_pred))

def calculate_num_disease_classes(rgb_dirs: list[str]) -> int:
    disease_classes = set()
    for directory in rgb_dirs:
        if os.path.isdir(directory):
            # Assuming that each directory represents a unique disease class
            disease_classes.update(next(os.walk(directory))[1])
    return len(disease_classes)


class ResNet50Model:
    def __init__(self, rgb_dirs: list[str], disease_segmented_dirs: list[str], leaf_segmented_dirs: list[str], learning_rate: float, val_split: float, dataset_name: str) -> None:
        self.rgb_dirs = rgb_dirs
        self.disease_segmented_dirs = disease_segmented_dirs
        self.leaf_segmented_dirs = leaf_segmented_dirs
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.num_disease_classes = calculate_num_disease_classes(self.rgb_dirs)
        self.dataset_name = dataset_name
        self.label_encoder = LabelEncoder()
        # Get all disease class names (folder names in the disease_segmented_dirs directory)
        all_labels = [label for label in os.listdir(self.disease_segmented_dirs) if os.path.isdir(os.path.join(self.disease_segmented_dirs, label))]
        self.label_encoder.fit(all_labels)
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
                
        #it the label encoder and transform class names to integer labels
        integer_encoded_disease_types = self.label_encoder.transform(disease_types)
        # convert integer labels to one-hot
        one_hot_encoded_disease_types = to_categorical(integer_encoded_disease_types, num_classes=self.num_disease_classes)

        # Normalize combined images for ResNet50
        combined_images = np.array(combined_images)  # No need to scale to [0, 255] range
        combined_images = preprocess_input(combined_images)  # Use ResNet50 preprocessing

        return np.array(combined_images), np.array(disease_masks), one_hot_encoded_disease_types
    
    def predict_and_save(self, paired_image_paths, output_dir, batch_size=32):
        if not os.path.exists(os.path.join(output_dir, 'images')):
            os.makedirs(os.path.join(output_dir, 'images'))

        # Initialize a list to keep track of prediction results
        prediction_results = []

        # Process images in batches
        for i in range(0, len(paired_image_paths), batch_size):
            batch = paired_image_paths[i:i + batch_size]
            batch_images = []

            for rgb_path, leaf_path, _, _ in batch:
                # Load RGB image
                rgb_image = img_to_array(load_img(rgb_path, target_size=(256, 256), color_mode='rgb')) / 255.0
                
                # Load leaf mask and expand dimensions if necessary
                leaf_mask = img_to_array(load_img(leaf_path, target_size=(256, 256), color_mode='grayscale')) / 255.0
                if leaf_mask.ndim == 2:
                    leaf_mask = np.expand_dims(leaf_mask, axis=-1)

                # Concatenate RGB image and leaf mask
                combined_image = np.concatenate([rgb_image, leaf_mask], axis=-1)
                batch_images.append(combined_image)

            batch_images = np.array(batch_images)
            preds = self.model.predict(batch_images)
            segmentation_masks = preds[0]  # Assuming this is the segmentation output
            classification_outputs = preds[1]  # Assuming this is the classification output

            # Process each image in the batch for results
            for j, (rgb_path, _, _, _) in enumerate(batch):
                # Save the segmentation mask
                mask_path = os.path.join(output_dir, 'images', f"mask_{i+j}.png")
                cv2.imwrite(mask_path, segmentation_masks[j, :, :, 0] * 255)

                # Get the class with the highest probability
                predicted_class = np.argmax(classification_outputs[j])

                # Store the classification result
                prediction_results.append({
                    'image_path': rgb_path,
                    'predicted_class': self.label_encoder.inverse_transform([predicted_class])[0],
                    'mask_path': mask_path
                })

        # Save the classification results to a CSV file
        results_df = pd.DataFrame(prediction_results)
        results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

        # Save the model
        self.model.save(os.path.join(output_dir, 'model.h5'))

    def _build_model(self) -> Model:
        # Load pre-trained ResNet50 model without top layers
        base_model = ResNet50(weights='imagenet', include_top=False)

        # Freeze pre-trained layers
        for layer in base_model.layers:
            layer.trainable = False

        # Create new input layer for 6-channel input
        input_tensor = Input(shape=(256, 256, 4))

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
        
        x_class = tf.keras.layers.GlobalAveragePooling2D()(x)
        x_class = tf.keras.layers.Dense(1024, activation='relu')(x_class)
        x_class = tf.keras.layers.Dropout(0.5)(x_class)
        disease_classification = tf.keras.layers.Dense(self.num_disease_classes, activation='softmax', name='disease_classification')(x_class)

        model = Model(inputs=input_tensor, outputs=[disease_segmentation, disease_classification])

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


        clr = CyclicLR(base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular')

        # Create EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1)

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
        
        def train_generator(data_generator, images, masks, labels):
            seed = 1  # Ensuring that image and mask undergo the same transformation
            image_gen = data_generator.flow(images, batch_size=batch_size, seed=seed)
            mask_gen = data_generator.flow(masks, batch_size=batch_size, seed=seed)

            original_labels = labels.copy()  # Save a copy of the original labels

            while True:
                image_batch = next(image_gen)
                mask_batch = next(mask_gen)
                # Get the corresponding labels for the batch
                label_batch = labels[:batch_size]
                # Remove the used labels
                labels = labels[batch_size:]
                # If the labels array is empty, reset it to the original labels
                if len(labels) == 0:
                    labels = original_labels.copy()
                yield image_batch, {'disease_segmentation': mask_batch, 'disease_classification': label_batch}


        train_gen = train_generator(augmenter, combined_inputs_train, disease_labels_train, train_disease_types)
        val_gen = train_generator(augmenter, combined_inputs_val, disease_labels_val, val_disease_types)

        # Model compilation
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), 
                    loss={'disease_segmentation': iou_loss,
                            'disease_classification': 'categorical_crossentropy'}, # you can use 'sparse_categorical_crossentropy' if your labels are integers
                    metrics={'disease_segmentation': ['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)],
                            'disease_classification': ['accuracy']},
                    loss_weights={'disease_segmentation': 1.0,
                                    'disease_classification': 0.5} # You may need to tune these weights
                    )

        # Model training
        history = self.model.fit(
            train_gen,
            steps_per_epoch=len(combined_inputs_train) // batch_size,
            validation_data=val_gen,
            validation_steps=len(combined_inputs_val) // batch_size,
            epochs=epochs,
            callbacks=[clr, early_stopping]
        )
        

        # Call predict_and_save on the validation images
        self.predict_and_save(stratified_val_data, output_dir)

        # Save the model and training history
        save_history_to_txt(history, output_dir)
        
        tf.keras.backend.clear_session()

        return history

