import os
import numpy as np
import tensorflow as tf

from keras.layers import Lambda, Input, Conv2D, MaxPooling2D, UpSampling2D, Resizing, concatenate, BatchNormalization, SpatialDropout2D, LeakyReLU, GlobalAveragePooling2D, Reshape, Dense, multiply, concatenate, SeparableConv2D
from keras.regularizers import l2
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import ReLU, Conv2DTranspose
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from utils.SaveHistoryToTxt import save_history_to_txt
from utils.BinarySegmentationMetrics import BinarySegmentationMetrics
from models.DL.DeepLearningUtils.ImagePreprocessing import pair_images_by_filename
from utils.CreateDirectory import _create_directory

def visualize_validation_data(inputs, labels, num_samples=5):
    """
    Visualize a sample of images and their corresponding labels from the validation set.

    Parameters:
    inputs (numpy.ndarray): The input images.
    labels (numpy.ndarray): The corresponding labels (masks).
    num_samples (int): Number of samples to visualize.
    """
    if num_samples > len(inputs):
        num_samples = len(inputs)
        print(f"Reducing num_samples to {len(inputs)} because it exceeds the number of available samples.")

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    for i in range(num_samples):
        ax = axes[i, 0]
        ax.imshow(inputs[i, ..., :3])  # Display RGB image
        ax.set_title("Input Image")
        ax.axis('off')

        ax = axes[i, 1]
        ax.imshow(labels[i].squeeze(), cmap='gray')  # Display the mask
        ax.set_title("True Label")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def save_predictions(model, inputs, labels, class_names, output_dir):
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all samples, not by class, to ensure all images are processed
    for idx in range(len(inputs)):
        input_img = inputs[idx]
        true_label = labels[idx]
        cls = labels[idx]  # Assuming labels are directly the class indices

        # Use the model to predict the label for this image
        pred_label = model.predict(np.expand_dims(input_img, axis=0))[0]

        # Apply threshold to convert probabilities to binary values
        pred_label_binary = (pred_label > 0.5).astype(np.float32)

        # Plot the input image, true label, and predicted label
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(input_img[..., :3])  # Only display the first 3 channels
        axs[0].set_title(f"Input Image - Class: {class_names[int(cls)]}")

        axs[1].imshow(true_label.squeeze(), cmap='gray')
        axs[1].set_title("True Label")

        axs[2].imshow(pred_label_binary.squeeze(), cmap='gray')
        axs[2].set_title("Predicted Binary Label")

        # Save the plot with a unique filename for each sample
        filename = f"class_{int(cls)}_sample_{idx}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close(fig)

def iou_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Intersection
    intersection = tf.reduce_sum(y_true * y_pred)
    # Union
    total = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    union = total - intersection
    # Adding a smooth factor to prevent division by 0
    smooth = 1e-6
    # Returning the IoU based loss
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou  # subtracting from 1 because we want to minimize the loss

def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return (lr * tf.math.exp(-0.1)).numpy()

def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice_loss = 1 - dice_coefficient(y_true, y_pred)
    iou = iou_loss(y_true, y_pred)
    return bce + dice_loss + iou * 2

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
        self.classes = sorted(os.listdir(rgb_dir))
        self.model = self._build_model()

    def _determine_num_classes(self, base_dir: str) -> int:
        """
        Determines the number of unique classes based on the number of subdirectories
        in a given directory. Assumes each subdirectory corresponds to a unique class.
        """
        return len([name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))])

    def load_images_and_masks(self, paired_image_paths, target_size=(256, 256)):
        combined_images = []
        disease_masks = []
        disease_types = [] 

        for rgb_path, disease_path, leaf_path, disease_type in paired_image_paths:
            # Load RGB image and normalize
            rgb_image = img_to_array(load_img(rgb_path, target_size=target_size, color_mode='rgb')) / 255.0
            # Load leaf mask and normalize
            leaf_mask = img_to_array(load_img(leaf_path, target_size=target_size, color_mode='grayscale')) / 255.0
            leaf_mask = np.expand_dims(leaf_mask[:,:,0], axis=-1)
            
            disease_mask = preprocess_mask(disease_path, target_size)

            combined_image = np.concatenate([rgb_image, leaf_mask], axis=-1)
            combined_images.append(combined_image)
            disease_masks.append(disease_mask)
            disease_types.append(disease_type) 

        return np.array(combined_images), np.array(disease_masks), np.array(disease_types)

    def _build_model(self) -> Model:
        input_tensor = Input(shape=(256, 256, 4))

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
        x = SpatialDropout2D(0.3)(x)  # Consider tuning the dropout rate
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=2)(x)

        # Continuing with Convolutional Layers and applying squeeze-excite blocks
        x = Conv2D(384, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        x = squeeze_excite_block(x)  # Optional: Test performance without this block

        x = Conv2D(384, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        x = squeeze_excite_block(x)  # Optional: Test performance without this block

        x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.01))(x)
        x = ReLU()(x)
        x = SpatialDropout2D(0.3)(x) # Consider tuning the dropout rate
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), strides=2)(x)

        # Upsampling
        x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
        x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
        x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
        x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)
        x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)

        # Resize x to match the dimensions of processed_mask
        x = Resizing(256, 256)(x) 

        # Combine features with the mask
        combined = concatenate([x, processed_mask])

        # Output layer for segmentation
        disease_segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='disease_segmentation')(combined)

        return Model(inputs=input_tensor, outputs=disease_segmentation)


    def compile_and_train(self, epochs: int, batch_size: int, output_dir: str) -> tf.keras.callbacks.History:
        # Directory setup
        _create_directory(output_dir)
        plots_dir = os.path.join(output_dir, 'plots')
        _create_directory(plots_dir)

        # Generate paired image paths with labels
        paired_image_paths_with_labels = pair_images_by_filename(self.rgb_dir, self.disease_segmented_dir, self.leaf_segmented_dir)

        # Split the data into training and validation sets while maintaining the structure
        train_data, val_data = train_test_split(paired_image_paths_with_labels, test_size=self.val_split, stratify=[item[3] for item in paired_image_paths_with_labels], random_state=42)

        # Load images and masks for training and validation
        combined_inputs_train, disease_labels_train, train_disease_types = self.load_images_and_masks(train_data, target_size=(256, 256))
        combined_inputs_val, disease_labels_val, val_disease_types = self.load_images_and_masks(val_data, target_size=(256, 256))

        # Initialize BinarySegmentationMetrics with validation data and disease types
        binary_segmentation_metrics = BinarySegmentationMetrics(validation_data=(combined_inputs_val, disease_labels_val), validation_disease_types=val_disease_types, model_name = 'AlexNet', learning_rate=self.learning_rate, val_split=self.val_split, dataset_name=self.dataset_name, output_dir=output_dir)
        
        print("Train input shape:", combined_inputs_train.shape)
        print("Validation input shape:", combined_inputs_val.shape)


        # Model compilation with binary segmentation in mind
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                        loss=combined_loss,
                        metrics=['accuracy', combined_loss])

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_combined_loss',  # Monitor validation loss
            factor=0.005,          # Reduce learning rate by a factor
            patience=10,         # Number of epochs with no improvement after which learning rate will be reduced
            verbose=1,           # Int verbosity level
            mode='auto',         # The direction is automatically inferred from the name of the monitored quantity
            min_delta=0.0001,    # A threshold for measuring the new optimum, to only focus on significant changes
            cooldown=0,          # Number of epochs to wait before resuming normal operation after lr has been reduced
            min_lr=0             # Lower bound on the learning rate
        )

        # Early stopping setup
        early_stopping = EarlyStopping(
            monitor='val_combined_loss',   # Monitor validation loss
            min_delta=0.001,      # Minimum change to qualify as an improvement
            patience=15,          # Number of epochs with no improvement after which training will be stopped
            verbose=1,            # Int verbosity level
            mode='auto',          # The direction is automatically inferred from the name of the monitored quantity
            baseline=None,        # Value for the monitored quantity. Training will stop if the model doesn't show improvement over this
            restore_best_weights=True  # Whether to restore model weights from the epoch with the best value of the monitored quantity
        )

        history = self.model.fit(
            combined_inputs_train, disease_labels_train,
            validation_data=(combined_inputs_val, disease_labels_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=[binary_segmentation_metrics, lr_scheduler, early_stopping]
        )
        
        if not os.path.exists(os.path.join(output_dir, 'predictions')):
            _create_directory(os.path.join(output_dir, 'predictions'))
        
        save_predictions(self.model, combined_inputs_val, disease_labels_val, val_disease_types, os.path.join(output_dir, 'predictions'))

        # Save training metrics and history
        save_history_to_txt(history, output_dir)
        
        binary_segmentation_metrics.save_results_to_excel()

        output_layer = self.model.layers[-1]
        print("Output layer type:", output_layer.__class__.__name__)
        print("Output layer config:", output_layer.get_config())
        
        tf.keras.backend.clear_session()

        return history




