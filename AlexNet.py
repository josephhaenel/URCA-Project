import os
import numpy as np
import tensorflow as tf
from F1Score import F1Score
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Resizing, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input  # Generic preprocessing

class AlexNetModel:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.model = self._build_model()

    @staticmethod
    def load_images(image_dir, is_mask=False, target_size=(227, 227)):
        images = []
        for filename in os.listdir(image_dir):
            if filename.endswith('.png'):
                image_path = os.path.join(image_dir, filename)
                image = load_img(image_path, target_size=target_size, color_mode='grayscale' if is_mask else 'rgb')
                image = img_to_array(image)
                if is_mask:
                    image = image / 255.0 
                else:
                    image = preprocess_input(image, mode='caffe') 
                images.append(image)
        return np.array(images)

    def _build_model(self):
        # Define the input
        input_tensor = Input(shape=(227, 227, 3))

        # AlexNet-like architecture with modifications for segmentation
        x = Conv2D(96, (11, 11), strides=4, activation='relu')(input_tensor)
        x = MaxPooling2D((3, 3), strides=2)(x)
        x = Conv2D(256, (5, 5), padding='same', activation='relu')(x)
        x = MaxPooling2D((3, 3), strides=2)(x)
        x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(384, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D((3, 3), strides=2)(x)

        # Upsampling layers for segmentation
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Resizing(256, 256)(x)

        # Final segmentation layers
        disease_segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='disease_segmentation')(x)
        leaf_segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='leaf_segmentation')(x)

        # Create the model
        model = Model(inputs=input_tensor, outputs=[disease_segmentation, leaf_segmentation])
        return model

    def compile_and_train(self, epochs, batch_size):
        images = self.load_images(os.path.join(self.base_dir, 'original'))
        disease_labels = self.load_images(os.path.join(self.base_dir, 'segmented_disease'), is_mask=True, target_size=(256, 256))
        segmentation_masks = self.load_images(os.path.join(self.base_dir, 'segmented_plant'), is_mask=True, target_size=(256, 256))

        f1_score_metric = F1Score()
        self.model.compile(optimizer='adam',
                           loss={'disease_segmentation': 'binary_crossentropy',
                                 'leaf_segmentation': 'binary_crossentropy'},
                           metrics=['accuracy', f1_score_metric, tf.keras.metrics.Recall()])

        history = self.model.fit(images, 
                                 {'disease_segmentation': disease_labels, 'leaf_segmentation': segmentation_masks},
                                 epochs=epochs,
                                 batch_size=batch_size)
        return history

if __name__ == '__main__':
    base_dir = 'tempdataset'
    alexnet_segmentation_model = AlexNetModel(base_dir)
    history = alexnet_segmentation_model.compile_and_train(epochs=10, batch_size=32)
