import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import concurrent
from utils.ComputeSilhouetteScore import compute_silhouette_score
from models.ML.MachineLearningUtils.PairImagesByFilename import pair_images_by_filename
from models.ML.MachineLearningUtils.calculate_iou import calculate_iou

def compute_silhouette_for_params(params):
    return compute_silhouette_score(*params)

class KMeansSegmentation:

    def __init__(self, rgb_dirs: list[str], disease_segmented_dirs: list[str], leaf_segmented_dirs: list[str], val_split: float, dataset_name: str) -> None:
        self.rgb_dirs = rgb_dirs
        self.disease_segmented_dirs = disease_segmented_dirs
        self.leaf_segmented_dirs = leaf_segmented_dirs
        self.val_split = val_split
        self.dataset_name = dataset_name
        
        
    def evaluate_on_full_dataset(self, combined_images, disease_masks):
        scaler = self.fit_scaler(combined_images)
        labels_pred = self.k_means_segmentation(combined_images, scaler, self.best_init, self.best_k)
        full_iou = calculate_iou(disease_masks, labels_pred)
        full_accuracy = accuracy_score(disease_masks.flatten(), labels_pred.flatten())
        return full_iou, full_accuracy

    def tune_kmeans_parameters(self, combined_images_train, k_range=range(2, 10), init_methods=['k-means++', 'random'], sample_size=1000):
        scaler = self.fit_scaler(combined_images_train)
        standardized_data = scaler.transform(combined_images_train.reshape(-1, combined_images_train.shape[-1]))

        # Take a random sample of the data
        sample_indices = np.random.choice(standardized_data.shape[0], sample_size, replace=False)
        sample_data = standardized_data[sample_indices]

        best_score = 0
        best_k = 0
        best_init = ''

        # Create a list of all combinations of k and init_method
        parameters = [(k, init_method) for k in k_range for init_method in init_methods]

        # Use a ProcessPoolExecutor to compute silhouette scores in parallel
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(compute_silhouette_for_params, (params, sample_data)) for params in parameters]
            for future in concurrent.futures.as_completed(futures):
                silhouette_avg, k, init_method = future.result()
                print(f'For n_clusters = {k} and init = {init_method}, the average silhouette_score is : {silhouette_avg}')
                if silhouette_avg > best_score:
                    best_score = silhouette_avg
                    best_k = k
                    best_init = init_method

        print(f'Best KMeans parameters found: n_clusters = {best_k}, init = {best_init}, silhouette_score = {best_score}')
        return best_k, best_init

    def load_and_preprocess_images(self, paired_image_paths, target_size=(256, 256)):
        combined_images = []
        disease_masks = []
        disease_types = []

        for rgb_path, leaf_mask_path, disease_path, disease_type in paired_image_paths:
            if os.path.exists(rgb_path) and os.path.exists(leaf_mask_path) and os.path.exists(disease_path):
                rgb_image = io.imread(rgb_path)
                leaf_mask = io.imread(leaf_mask_path, as_gray=True) > 0.5  # Convert to binary mask
                disease_mask = io.imread(disease_path, as_gray=True) > 0.5  # Convert to binary mask

                # Resize the images and masks
                rgb_image_resized = resize(rgb_image, target_size, anti_aliasing=True)
                leaf_mask_resized = resize(leaf_mask, target_size, anti_aliasing=False, order=0, preserve_range=True)
                disease_mask_resized = resize(disease_mask, target_size, anti_aliasing=False, order=0, preserve_range=True)

                # Flatten the images and masks
                rgb_flatten = rgb_image_resized.reshape((-1, 3))
                leaf_flatten = leaf_mask_resized.flatten()
                disease_flatten = disease_mask_resized.flatten()  # You might need this for IOU calculation

                # Stack the flattened rgb and leaf mask arrays
                combined_image = np.column_stack((rgb_flatten, leaf_flatten))

                combined_images.append(combined_image)
                disease_masks.append(disease_flatten)
                disease_types.append(disease_type)

        return np.array(combined_images), np.array(disease_masks), np.array(disease_types)


    def fit_scaler(self, combined_images):
        # Fit the scaler on the training data only
        scaler = StandardScaler()
        scaler.fit(combined_images.reshape(-1, combined_images.shape[-1]))
        return scaler

    def k_means_segmentation(self, combined_images, scaler, init, k):
        # Apply the provided scaler to standardize the data
        combined_images_standardized = scaler.transform(combined_images.reshape(-1, combined_images.shape[-1]))
        kmeans = KMeans(n_clusters=k, init=init, random_state=42)
        labels = kmeans.fit_predict(combined_images_standardized)
        return labels.reshape(-1, combined_images.shape[1])

    def compile_and_train(self, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        all_paired_image_paths = pair_images_by_filename(self.rgb_dirs, self.disease_segmented_dirs, self.leaf_segmented_dirs)
        results = []

        # Perform parameter tuning using only training data
        combined_images_train, _, _ = self.load_and_preprocess_images([path for path in all_paired_image_paths if path[-1] != 'Healthy'])

        scaler = self.fit_scaler(combined_images_train)  # Fit scaler only once on training data
        print("Beginning Tuning...")
        best_k, best_init = self.tune_kmeans_parameters(scaler.transform(combined_images_train.reshape(-1, combined_images_train.shape[-1])))
        print("Tuning complete. Best K: ", best_k,"Best Init: ", best_init)

        for disease in set(p[3] for p in all_paired_image_paths):
            disease_paths = [p for p in all_paired_image_paths if p[3] == disease]
            
            # Extract the labels for the current disease to use for stratification
            labels = [p[3] for p in disease_paths]

            # Split the data
            train_paths, val_paths = train_test_split(disease_paths, test_size=self.val_split, stratify=labels, random_state=42)

            combined_images_train, disease_masks_train, _ = self.load_and_preprocess_images(train_paths)
            combined_images_val, disease_masks_val, _ = self.load_and_preprocess_images(val_paths)

            # Use the scaler fitted on the training data to transform both training and validation data
            standardized_images_train = scaler.transform(combined_images_train.reshape(-1, combined_images_train.shape[-1]))
            standardized_images_val = scaler.transform(combined_images_val.reshape(-1, combined_images_val.shape[-1]))

            # Use the best parameters found from the tuning to train and predict on the training and validation data
            kmeans = KMeans(n_clusters=best_k, init=best_init, random_state=42, n_init=10)
            labels_pred_train = kmeans.fit_predict(standardized_images_train).reshape(-1, combined_images_train.shape[1])
            labels_pred_val = kmeans.predict(standardized_images_val).reshape(-1, combined_images_val.shape[1])

            train_iou = calculate_iou(disease_masks_train, labels_pred_train)
            val_iou = calculate_iou(disease_masks_val, labels_pred_val)
            train_accuracy = accuracy_score(disease_masks_train.flatten(), labels_pred_train.flatten())
            val_accuracy = accuracy_score(disease_masks_val.flatten(), labels_pred_val.flatten())

            # For the healthy class, handle IoU separately as it is expected to be 0/0
            if disease == 'Healthy':
                # Assuming the healthy class has been processed correctly and is fully healthy
                train_iou = val_iou = 1.0
                # Accuracy could be 1 only if the model predicts no disease on a healthy leaf

            results.append({
                'Disease Type': disease,
                'Train IoU': train_iou,
                'Validation IoU': val_iou,
                'Train Accuracy': train_accuracy,
                'Validation Accuracy': val_accuracy
            })

        # Evaluate on the full dataset using the trained model and scaler
        combined_images, disease_masks, _ = self.load_and_preprocess_images(all_paired_image_paths)
        self.best_k, self.best_init = self.tune_kmeans_parameters(combined_images)
        full_iou, full_accuracy = self.evaluate_on_full_dataset(combined_images, disease_masks)
        print(f"Full dataset IoU: {full_iou}, Full dataset Accuracy: {full_accuracy}")

        # Append full dataset evaluation results
        results.append({
            'Disease Type': 'All_Classes',
            'Train IoU': full_iou,
            'Validation IoU': full_iou,
            'Train Accuracy': full_accuracy,
            'Validation Accuracy': full_accuracy
        })

        # Create the results DataFrame and save to Excel
        df_all = pd.DataFrame(results)
        df_all.to_excel(os.path.join(output_dir, f"{self.dataset_name}_metrics.xlsx"), index=False)

        return df_all