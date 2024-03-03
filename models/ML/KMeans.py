import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from skimage import io, color
from skimage.transform import resize
from utils.BinarySegmentationMetrics import BinarySegmentationMetrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

class KMeansSegmentation:

    def __init__(self, rgb_dirs: list[str], disease_segmented_dirs: list[str], leaf_segmented_dirs: list[str], val_split: float, dataset_name: str) -> None:
        self.rgb_dirs = rgb_dirs
        self.disease_segmented_dirs = disease_segmented_dirs
        self.leaf_segmented_dirs = leaf_segmented_dirs
        self.val_split = val_split
        self.dataset_name = dataset_name
        
    def calculate_iou(self, y_true, y_pred):
        intersection = np.logical_and(y_true, y_pred)
        union = np.logical_or(y_true, y_pred)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    def load_and_preprocess_images(self, paired_image_paths, target_size=(299, 299)):
        combined_images = []
        disease_masks = []
        disease_types = []

        for rgb_path, leaf_mask_path, disease_path, disease_type in paired_image_paths:
            if os.path.exists(rgb_path) and os.path.exists(leaf_mask_path) and os.path.exists(disease_path):
                rgb_image = io.imread(rgb_path)
                leaf_mask = io.imread(leaf_mask_path, as_gray=True)
                disease_mask = io.imread(disease_path, as_gray=True)

                # Resize images to the target size
                rgb_image = resize(rgb_image, target_size, anti_aliasing=False)
                leaf_mask = resize(leaf_mask, target_size, anti_aliasing=False)
                disease_mask = resize(
                    disease_mask, target_size, anti_aliasing=False)

                # Flatten the images for k-means clustering
                rgb_flatten = rgb_image.reshape((-1, 3))
                leaf_flatten = leaf_mask.flatten()
                combined_image = np.column_stack((rgb_flatten, leaf_flatten))

                combined_images.append(combined_image)
                disease_masks.append(disease_mask.flatten())
                disease_types.append(disease_type)

        return np.array(combined_images), np.array(disease_masks), np.array(disease_types)
    
    def calculate_iou_per_class(self, y_true, y_pred):
        num_classes = np.unique(y_true).shape[0]
        iou_per_class = []
        for i in range(num_classes):
            temp_true = np.where(y_true == i, 1, 0)
            temp_pred = np.where(y_pred == i, 1, 0)
            iou = np.sum(temp_true * temp_pred) / np.sum(temp_true + temp_pred - temp_true * temp_pred)
            iou_per_class.append(iou)
        return iou_per_class

    def calculate_accuracy_per_class(self, y_true, y_pred):
        num_classes = np.unique(y_true).shape[0]
        accuracy_per_class = []
        for i in range(num_classes):
            temp_true = np.where(y_true == i, 1, 0)
            temp_pred = np.where(y_pred == i, 1, 0)
            accuracy = np.sum(temp_true == temp_pred) / np.prod(y_true.shape)
            accuracy_per_class.append(accuracy)
        return accuracy_per_class

    def pair_images_by_filename(self, base_rgb_dir: str, base_disease_dir: str, base_leaf_dir: str) -> list[tuple[str, str, str, str]]:
        paired_images = []
        for disease in os.listdir(base_rgb_dir):
            rgb_dir = os.path.join(base_rgb_dir, disease)
            disease_dir = os.path.join(base_disease_dir, disease)
            leaf_dir = os.path.join(base_leaf_dir, disease)

            if not os.path.isdir(rgb_dir) or not os.path.isdir(disease_dir) or not os.path.isdir(leaf_dir):
                print(
                    f"One of the directories is invalid: {rgb_dir}, {disease_dir}, {leaf_dir}")
                continue

            for file_name in os.listdir(rgb_dir):
                if file_name.endswith('.png'):
                    rgb_path = os.path.join(rgb_dir, file_name)
                    disease_path = os.path.join(disease_dir, file_name)
                    leaf_path = os.path.join(leaf_dir, file_name)

                    if os.path.exists(rgb_path) and os.path.exists(disease_path) and os.path.exists(leaf_path):
                        paired_images.append(
                            (rgb_path, leaf_path, disease_path, disease))
                    else:
                        print(f"Missing image for {file_name} in {disease}")

        return paired_images

    def _create_directory(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)

    def print_history_keys(self, history: dict) -> None:
        print("Keys in training history:")
        for key in history.keys():
            print(key)


    def k_means_segmentation(self, combined_images):
        # Standardize the data
        scaler = StandardScaler()
        combined_images_standardized = scaler.fit_transform(
            combined_images.reshape(-1, combined_images.shape[-1]))

        # Apply k-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        disease_labels_pred = kmeans.fit_predict(combined_images_standardized)

        # Reshape the predicted labels to match the original image shape
        disease_labels_pred = disease_labels_pred.reshape(
            combined_images.shape[0], -1)

        return disease_labels_pred

    def compile_and_train(self, output_dir: str):
        self._create_directory(output_dir)

        # Load and preprocess data
        all_paired_image_paths = self.pair_images_by_filename(
            self.rgb_dirs, self.disease_segmented_dirs, self.leaf_segmented_dirs)

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
                stratify=[p[3] for p in paths],
                random_state=42
            )
            stratified_train_data.extend(train_paths)
            stratified_val_data.extend(val_paths)

        # Preparing training and validation datasets
        combined_inputs_train, disease_labels_train, train_disease_types = self.load_and_preprocess_images(
            stratified_train_data)
        combined_inputs_val, disease_labels_val, val_disease_types = self.load_and_preprocess_images(
            stratified_val_data)

        # K-means segmentation
        disease_labels_pred_train = self.k_means_segmentation(
            combined_inputs_train)
        disease_labels_pred_val = self.k_means_segmentation(
            combined_inputs_val)

        # Evaluate the performance
        train_accuracy = accuracy_score(
            disease_labels_train.flatten(), disease_labels_pred_train.flatten())
        val_accuracy = accuracy_score(
            disease_labels_val.flatten(), disease_labels_pred_val.flatten())
        
        train_iou = self.calculate_iou(disease_labels_train, disease_labels_pred_train)
        val_iou = self.calculate_iou(disease_labels_val, disease_labels_pred_val)
        
        train_iou_per_class = self.calculate_iou_per_class(disease_labels_train, disease_labels_pred_train)
        val_iou_per_class = self.calculate_iou_per_class(disease_labels_val, disease_labels_pred_val)
        train_accuracy_per_class = self.calculate_accuracy_per_class(disease_labels_train, disease_labels_pred_train)
        val_accuracy_per_class = self.calculate_accuracy_per_class(disease_labels_val, disease_labels_pred_val)
        
        df = pd.DataFrame({'Class': range(len(train_iou_per_class)),
            'Train IoU': train_iou_per_class,
            'Validation IoU': val_iou_per_class,
            'Train Accuracy': train_accuracy_per_class,
            'Validation Accuracy': val_accuracy_per_class
        })
        
        df.to_excel('metrics_per_class.xlsx', index=False)

        print(f'Training Accuracy: {train_accuracy}')
        print(f'Training IoU: {train_iou}')
        print(f'Validation IoU: {val_iou}')
        print(f'Validation Accuracy: {val_accuracy}')

        return {'train_accuracy': train_accuracy, 'val_accuracy': val_accuracy, 'val_iou' : val_iou}

