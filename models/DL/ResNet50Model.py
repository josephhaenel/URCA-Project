import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, Conv2D, UpSampling2D, Resizing
from tensorflow.keras.layers import Concatenate, BatchNormalization, Activation
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from utils.F1Score import F1Score
from utils.GraphPlotter import save_plots, save_history_to_txt
from tensorflow.keras.metrics import Recall
from utils.IoUMetric import IoUMetric

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

        # Compute IoU
        iou_score = intersection / union
        return iou_score



class ResNet50Model:
    def __init__(self, rgb_dirs, disease_segmented_dirs, leaf_segmented_dirs):
        self.rgb_dirs = rgb_dirs
        self.disease_segmented_dirs = disease_segmented_dirs
        self.leaf_segmented_dirs = leaf_segmented_dirs
        self.model = self._build_model()

    def load_images_and_masks(self, paired_image_paths, target_size=(224, 224)):
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
    
    def load_images(self, image_paths, is_mask: bool = False, target_size: tuple = (224, 224)) -> np.ndarray:
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

    def _build_model(self):
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
        leaf_segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='leaf_segmentation')(x)

        model = Model(inputs=base_model.input, outputs=[disease_segmentation, leaf_segmentation])
        return model

    def _create_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def compile_and_train(self, epochs, batch_size, output_dir, validation_split = 0.5):
        # Directory setup
        self._create_directory(output_dir)

        # Load and preprocess data
        paired_image_paths = self.pair_images_by_filename(self.rgb_dirs, self.disease_segmented_dirs, self.leaf_segmented_dirs)
        combined_inputs = self.load_images_and_masks(paired_image_paths)
        disease_labels = self.load_images([d[2] for d in paired_image_paths], is_mask=True, target_size=(256, 256))
        leaf_labels = self.load_images([d[1] for d in paired_image_paths], is_mask=True, target_size=(256, 256))
        
        total_size = len(combined_inputs)
        training_size = int(total_size * (1 - validation_split))
        validation_size = total_size - training_size
        
        print(f"Total dataset size: {total_size}")
        print(f"Training dataset size: {training_size}")
        print(f"Validation dataset size: {validation_size}")

        # Model compilation
        disease_metrics = [BinaryCrossentropy(), 'accuracy', F1Score(), Recall(name='recall'), IoUMetric()]
        leaf_metrics = [BinaryCrossentropy(), 'accuracy', F1Score(), Recall(name='recall')]

        self.model.compile(
            optimizer=Adam(learning_rate=0.0000005),
            loss={'disease_segmentation': BinaryCrossentropy(), 'leaf_segmentation': BinaryCrossentropy()},
            metrics={'disease_segmentation': disease_metrics, 'leaf_segmentation': leaf_metrics}
        )

        # Model training
        history = self.model.fit(
            combined_inputs, 
            {'disease_segmentation': disease_labels, 'leaf_segmentation': leaf_labels},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )

        # Calculate and Print IoU Scores
        predictions = self.model.predict(combined_inputs)
        predictions = self.model.predict(combined_inputs)
        iou_score = CalculateIOU.calculate_iou(disease_labels, predictions[0])
        print(f"Validation IoU for disease segmentation: {iou_score.numpy()}")

        # Save training metrics and history
        save_history_to_txt(history, output_dir)

        return history

