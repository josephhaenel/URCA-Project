import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, Conv2D, MaxPooling2D, UpSampling2D, Resizing, concatenate, BatchNormalization, SpatialDropout2D, Conv2DTranspose, Dropout, LeakyReLU, GlobalAveragePooling2D, Reshape, Dense, multiply, concatenate, SeparableConv2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.metrics import Recall
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils.F1Score import F1Score
from utils.GraphPlotter import save_history_to_txt
from utils.IoUMetric import IoUMetric, IoULogger
from utils.LogPerClassMetrics import LogPerClassMetrics
from utils.BinaryPerClassMetrics import BinaryPerClassMetrics
from utils.BinarySegmentationMetrics import BinarySegmentationMetrics
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return (lr * tf.math.exp(-0.1)).numpy()

def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice_loss = 1 - dice_coefficient(y_true, y_pred)
    return bce + dice_loss

def preprocess_mask(mask_path, target_size):
    # Load the mask
    mask = img_to_array(load_img(mask_path, target_size=target_size, color_mode='grayscale'))
    # Convert to binary format: 1 if disease is present, 0 otherwise
    binary_mask = np.where(mask > 0, 1, 0)
    return binary_mask[..., np.newaxis]  # Add a channel dimension

def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = multiply([init, se])
    return x

class AlexNetModel:
    def __init__(self, rgb_dir: str, disease_segmented_dir: str, leaf_segmented_dir: str,
                 learning_rate: float, val_split: float, dataset_name: str) -> None:
        self.rgb_dir = rgb_dir
        self.disease_segmented_dir = disease_segmented_dir
        self.leaf_segmented_dir = leaf_segmented_dir
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.dataset_name = dataset_name
        self.num_classes = self._determine_num_classes(disease_segmented_dir)
        self.model = self._build_model()

    def _determine_num_classes(self, base_dir: str) -> int:
        """
        Determines the number of unique classes based on the number of subdirectories
        in a given directory. Assumes each subdirectory corresponds to a unique class.
        """
        return len([name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))])
        
    def pair_images_by_filename(self, base_rgb_dir: str, base_disease_dir: str, base_leaf_dir: str):
        paired_images_with_labels = []
        # Loop through each disease type directory in the RGB directory
        for disease_type in os.listdir(base_rgb_dir):
            rgb_disease_dir = os.path.join(base_rgb_dir, disease_type)
            disease_segmented_disease_dir = os.path.join(base_disease_dir, disease_type)
            leaf_segmented_disease_dir = os.path.join(base_leaf_dir, disease_type)

            # Check that directories exist in all three locations
            if not os.path.isdir(rgb_disease_dir) or not os.path.isdir(disease_segmented_disease_dir) or not os.path.isdir(leaf_segmented_disease_dir):
                continue

            for file_name in os.listdir(rgb_disease_dir):
                rgb_path = os.path.join(rgb_disease_dir, file_name)
                disease_path = os.path.join(disease_segmented_disease_dir, file_name)
                leaf_path = os.path.join(leaf_segmented_disease_dir, file_name)

                # Ensure all paths exist before adding
                if os.path.exists(rgb_path) and os.path.exists(disease_path) and os.path.exists(leaf_path):
                    paired_images_with_labels.append((rgb_path, disease_path, leaf_path, disease_type))
                else:
                    print(f"Missing image for {file_name} in {disease_type}")
        
        return paired_images_with_labels

    def load_images_and_masks(self, paired_image_paths, target_size=(227, 227)):
        combined_images = []
        disease_masks = []
        disease_types = [] 

        for rgb_path, disease_path, leaf_path, disease_type in paired_image_paths:
            rgb_image = img_to_array(load_img(rgb_path, target_size=target_size, color_mode='rgb')) / 255.0
            leaf_mask = img_to_array(load_img(leaf_path, target_size=target_size, color_mode='grayscale')) / 255.0
            leaf_mask = np.repeat(leaf_mask, 3, axis=-1)
            disease_mask = preprocess_mask(disease_path, target_size)

            combined_image = np.concatenate([rgb_image, leaf_mask], axis=-1)
            combined_images.append(combined_image)
            disease_masks.append(disease_mask)
            disease_types.append(disease_type) 

        return np.array(combined_images), np.array(disease_masks), np.array(disease_types)


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

    def _build_model(self) -> Model:
        input_tensor = Input(shape=(227, 227, 6))

        # Extract RGB and mask channels from the input
        processed_rgb = Lambda(lambda x: x[:, :, :, :3])(input_tensor)
        processed_mask = Lambda(lambda x: x[:, :, :, 3:])(input_tensor)

        # Initial Convolution and Pooling Layers
        x = Conv2D(96, (11, 11), strides=4, kernel_regularizer=l2(0.01))(processed_rgb)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=2)(x)

        # Second Convolution with depthwise separable convolution
        x = SeparableConv2D(256, (5, 5), padding='same', depthwise_regularizer=l2(0.01), pointwise_regularizer=l2(0.01))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = SpatialDropout2D(0.5)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=2)(x)

        # Continuing with Convolutional Layers and applying squeeze-excite blocks
        x = Conv2D(384, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        x = squeeze_excite_block(x)

        x = Conv2D(384, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        x = squeeze_excite_block(x)

        x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = SpatialDropout2D(0.5)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=2)(x)

        # Upsampling
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Resizing(227, 227)(x)

        # Combine features with the mask
        combined = concatenate([x, processed_mask])

        # Output layer for segmentation
        disease_segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='disease_segmentation')(combined)


        return Model(inputs=input_tensor, outputs=disease_segmentation)


    def _create_directory(self, path: str) -> None:

        if not os.path.exists(path):
            os.makedirs(path)

    def compile_and_train(self, epochs: int, batch_size: int, output_dir: str) -> tf.keras.callbacks.History:
        # Directory setup
        self._create_directory(output_dir)
        plots_dir = os.path.join(output_dir, 'plots')
        self._create_directory(plots_dir)

        # Generate paired image paths with labels
        paired_image_paths_with_labels = self.pair_images_by_filename(self.rgb_dir, self.disease_segmented_dir, self.leaf_segmented_dir)

        # Split the data into training and validation sets while maintaining the structure
        train_data, val_data = train_test_split(paired_image_paths_with_labels, test_size=self.val_split, stratify=[item[3] for item in paired_image_paths_with_labels], random_state=42)

        # Load images and masks for training and validation
        combined_inputs_train, disease_labels_train, train_disease_types = self.load_images_and_masks(train_data, target_size=(227, 227))
        combined_inputs_val, disease_labels_val, val_disease_types = self.load_images_and_masks(val_data, target_size=(227, 227))

        # Initialize BinarySegmentationMetrics with validation data and disease types
        binary_segmentation_metrics = BinarySegmentationMetrics(validation_data=(combined_inputs_val, disease_labels_val), validation_disease_types=val_disease_types, model_name = 'AlexNet', learning_rate=self.learning_rate, val_split=self.val_split, dataset_name=self.dataset_name, output_dir=output_dir)

        # Model compilation with binary segmentation in mind
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

        lr_callback = LearningRateScheduler(scheduler)

        # EarlyStopping callback
        es_callback = EarlyStopping(monitor='val_accuracy', patience=5)

        history = self.model.fit(
            combined_inputs_train, disease_labels_train,
            validation_data=(combined_inputs_val, disease_labels_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=[binary_segmentation_metrics, lr_callback, es_callback]
        )

        # Save training metrics and history
        save_history_to_txt(history, output_dir)
        
        binary_segmentation_metrics.save_results_to_excel()

        output_layer = self.model.layers[-1]
        print("Output layer type:", output_layer.__class__.__name__)
        print("Output layer config:", output_layer.get_config())

        return history




