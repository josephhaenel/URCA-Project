import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, recall_score, precision_score
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
    
    @staticmethod
    def convert_to_binary_mask(image):
        # Convert the image to grayscale (Should be already)
        grayscale = np.mean(image, axis=-1) if image.ndim == 3 else image
        # Convert to binary mask (1 for white, 0 for black)
        binary_mask = (grayscale > 0.5).astype(int)
        return binary_mask

    def apply_kmeans(self, image):
        w, h, d = tuple(image.shape)
        image_flattened = np.reshape(image, (w * h, d))
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=0).fit(image_flattened)
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

    def process_folder(self, folder_path):
        original_folder = os.path.join(folder_path, 'original')
        disease_folder = os.path.join(folder_path, 'segmented_disease')
        metrics = {'accuracy': [], 'recall': [], 'precision': []}

        for image_name in os.listdir(original_folder):
            original_image_path = os.path.join(original_folder, image_name)
            disease_image_path = os.path.join(disease_folder, image_name)

            original_image = self.load_and_preprocess_image(original_image_path)
            disease_image = self.load_and_preprocess_image(disease_image_path)

            labels = self.apply_kmeans(original_image)
            
            # Optional if you want top see what the segmentation looks like
            self.visualize_segmentation(original_image, labels)

            ground_truth = self.convert_to_binary_mask(disease_image).flatten()
            predicted = labels.flatten()

            if ground_truth.size == predicted.size:
                metrics['accuracy'].append(accuracy_score(ground_truth, predicted))
                metrics['recall'].append(recall_score(ground_truth, predicted, average='macro', zero_division=1))
                metrics['precision'].append(precision_score(ground_truth, predicted, average='macro'))
            else:
                print(f"Size mismatch in image {image_name}: ground truth {ground_truth.size}, predicted {predicted.size}") # Used for debugging

        return metrics



if __name__ == '__main__':
    folder_path = 'tempdataset'
    n_clusters = 5  # May need to modify as needed

    segmenter = ImageKMeansSegmenter(n_clusters)
    results = segmenter.process_folder(folder_path)

    # Output the average of the metrics
    print("Average Accuracy:", np.mean(results['accuracy']))
    print("Average Recall:", np.mean(results['recall']))
    print("Average Precision:", np.mean(results['precision']))
