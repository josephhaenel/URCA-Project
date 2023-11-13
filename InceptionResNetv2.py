import os
import numpy as np
import tensorflow as tf
from F1Score import F1Score
from tensorflow.keras.layers import Resizing
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load and preprocess images
def load_images(image_dir, is_mask=False, target_size=(299, 299)):
    images = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            image = load_img(image_path, target_size=target_size, color_mode='grayscale' if is_mask else 'rgb')
            image = img_to_array(image)
            if is_mask:
                image = image / 255.0  # Normalize mask images
            else:
                image = preprocess_input(image)  # Preprocess only non-mask images
            images.append(image)
    return np.array(images)

# Dataset directories
base_dir = 'tempdataset'
original_dir = os.path.join(base_dir, 'original')
segmented_disease_dir = os.path.join(base_dir, 'segmented_disease')
segmented_leaf_dir = os.path.join(base_dir, 'segmented_plant')

# Load images and masks
images = load_images(original_dir)
disease_labels = load_images(segmented_disease_dir, is_mask=True, target_size=(256, 256))  
segmentation_masks = load_images(segmented_leaf_dir, is_mask=True, target_size=(256, 256))  

# Load the base InceptionResNetV2 model
base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))

# For segmentation
x = base_model.output

# Add convolution and upsampling layers
x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D(size=(2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D(size=(2, 2))(x)

# Resize to the exact target size
x = Resizing(256, 256)(x)

# Final layers for segmentation tasks
disease_segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='disease_segmentation')(x)
leaf_segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='leaf_segmentation')(x)

# Define the model
model = Model(inputs=base_model.input, outputs=[disease_segmentation, leaf_segmentation])

# Define the F1Score metric
f1_score_metric = F1Score()

# Compile the model with appropriate loss functions, metrics, and F1 score
model.compile(optimizer='adam',
              loss={'disease_segmentation': 'binary_crossentropy',
                    'leaf_segmentation': 'binary_crossentropy'},
              metrics=['accuracy', f1_score_metric, tf.keras.metrics.Recall()]
              )

# Train the model
history = model.fit(
    images, 
    {'disease_segmentation': disease_labels, 'leaf_segmentation': segmentation_masks},
    epochs=10,
    batch_size=32
)

