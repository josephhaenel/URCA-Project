import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class ImageRandomForestSegmenter:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=0)  # You can adjust parameters as needed

    @staticmethod
    def load_and_preprocess_image(image_path, target_size=(227, 227)):
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image)
        image = np.array(image, dtype=np.float64) / 255
        return image

    @staticmethod
    def convert_to_binary_mask(image):
        # Convert the image to grayscale (Should be already)
        grayscale = np.mean(image, axis=-1) if image.ndim == 3 else image
        # Convert to binary mask (1 for white, 0 for black)
        binary_mask = (grayscale > 0.5).astype(int)
        return binary_mask

    def train_model(self, training_data, training_labels):
        self.model.fit(training_data, training_labels)

    def predict(self, image):
        w, h, d = tuple(image.shape)
        image_flattened = np.reshape(image, (w * h, d))
        labels = self.model.predict(image_flattened)
        return labels.reshape(w, h)

    def visualize_segmentation(self, original_image, labels):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(labels, cmap='viridis')
        plt.title('Image Segmentation')
        plt.show()

    def process_folder(self, folder_path):
        original_folder = os.path.join(folder_path, 'original')
        disease_folder = os.path.join(folder_path, 'segmented_disease')
        metrics = {'accuracy': [], 'recall': [], 'precision': []}

        # TODO: Split data into training and validation set
        # self.train_model(training_data, training_labels)

        for image_name in os.listdir(original_folder):
            original_image_path = os.path.join(original_folder, image_name)
            disease_image_path = os.path.join(disease_folder, image_name)

            original_image = self.load_and_preprocess_image(original_image_path)
            disease_image = self.load_and_preprocess_image(disease_image_path)

            labels = self.predict(original_image)

            # self.visualize_segmentation(original_image, labels) optional

            ground_truth = self.convert_to_binary_mask(disease_image).flatten()
            predicted = labels.flatten()

            # Not done yet

        return metrics

# TODO: need to handle the splitting of dataset into training and test sets
