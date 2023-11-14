import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class ImageKMeansSegmenter:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters

    @staticmethod
    def load_and_preprocess_image(image_path, target_size=(227, 227)):
        image = load_img(image_path, target_size=target_size)
        image = img_to_array(image)
        image = np.array(image, dtype=np.float64) / 255
        return image

    def apply_kmeans(self, image):
        w, h, d = tuple(image.shape)
        image_flattened = np.reshape(image, (w * h, d))
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(image_flattened)
        labels = kmeans.labels_
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

# Example Usage
image_path = 'path_to_your_image.jpg'
n_clusters = 5  # Modify as needed

segmenter = ImageKMeansSegmenter(n_clusters)
original_image = segmenter.load_and_preprocess_image(image_path)
labels = segmenter.apply_kmeans(original_image)
segmenter.visualize_segmentation(original_image, labels)
