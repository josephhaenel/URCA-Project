import os
import numpy as np
import tensorflow as tf
from F1Score import F1Score
from tensorflow.keras.layers import Resizing, Input, Conv2D, UpSampling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class ResNet50Model:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.model = self._build_model()

    @staticmethod
    def load_images(image_dir, is_mask=False, target_size=(224, 224)):
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

    def _build_model(self):
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

        x = base_model.output
        x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D(size=(2, 2))(x)
        x = Resizing(256, 256)(x)

        disease_segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='disease_segmentation')(x)
        leaf_segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='leaf_segmentation')(x)

        model = Model(inputs=base_model.input, outputs=[disease_segmentation, leaf_segmentation])
        return model

    def compile_and_train(self, epochs, batch_size):
        images = self.load_images(os.path.join(self.base_dir, 'original'), target_size=(224, 224))
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
    resnet50_segmentation_model = ResNet50Model(base_dir)
    history = resnet50_segmentation_model.compile_and_train(epochs=10, batch_size=32)
