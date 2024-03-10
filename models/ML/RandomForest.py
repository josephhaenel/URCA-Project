import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import pandas as pd
from models.ML.MachineLearningUtils.PairImagesByFilename import pair_images_by_filename
from models.ML.MachineLearningUtils.calculate_iou import calculate_iou
import multiprocessing

class ImageRandomForestSegmenter:
    """
    A class that performs image segmentation using random forest.

    Args:
        rgb_dirs (list[str]): List of directories containing RGB images.
        disease_segmented_dirs (list[str]): List of directories containing disease segmented images.
        leaf_segmented_dirs (list[str]): List of directories containing leaf segmented images.
        val_split (float): The proportion of data to be used for validation.
        dataset_name (str): The name of the dataset.

    Attributes:
        rgb_dirs (list[str]): List of directories containing RGB images.
        disease_segmented_dirs (list[str]): List of directories containing disease segmented images.
        leaf_segmented_dirs (list[str]): List of directories containing leaf segmented images.
        val_split (float): The proportion of data to be used for validation.
        dataset_name (str): The name of the dataset.

    Methods:
        load_and_preprocess_images: Load and preprocess the images.
        fit_scaler: Fit a StandardScaler on the training data.
        random_forest_classification: Perform random forest classification on the provided data.
        compile_and_train: Compile and train a random forest model for image classification.
    """
    def __init__(self, rgb_dirs: list[str], disease_segmented_dirs: list[str], leaf_segmented_dirs: list[str], val_split: float, dataset_name: str) -> None:
        self.rgb_dirs = rgb_dirs
        self.disease_segmented_dirs = disease_segmented_dirs
        self.leaf_segmented_dirs = leaf_segmented_dirs
        self.val_split = val_split
        self.dataset_name = dataset_name

    def load_and_preprocess_images(self, paired_image_paths, target_size=(256, 256)):
        """
        Load and preprocess the images.

        Args:
            paired_image_paths (list): List of paired image paths.
            target_size (tuple): The target size for resizing the images.

        Returns:
            tuple: A tuple containing the combined images, disease masks, and disease types.
        """
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
        """
        Fits a StandardScaler on the training data.

        Parameters:
        combined_images (numpy.ndarray): The combined images used for fitting the scaler.

        Returns:
        StandardScaler: The fitted scaler object.
        """
        scaler = StandardScaler()
        scaler.fit(combined_images.reshape(-1, combined_images.shape[-1]))
        return scaler
    
    def random_forest_classification(self, combined_images, disease_masks, scaler):
        """
        Perform random forest classification on the provided data.

        Parameters:
        - combined_images (numpy.ndarray): The combined images data.
        - disease_masks (numpy.ndarray): The disease masks data.
        - scaler (sklearn.preprocessing.StandardScaler): The scaler used to standardize the data.

        Returns:
        - RandomForestClassifier: The trained random forest classifier.
        """
        num_cores = multiprocessing.cpu_count()
        # Apply the provided scaler to standardize the data
        combined_images_standardized = scaler.transform(combined_images.reshape(-1, combined_images.shape[-1]))
        # Initialize RandomForestClassifier with n_jobs=-1 to use all CPU cores
        rf = RandomForestClassifier(random_state=42, n_jobs=int((1/2) * num_cores))
        # Train the RandomForestClassifier using the disease masks as labels
        rf.fit(combined_images_standardized, disease_masks.flatten())
        return rf

    def compile_and_train(self, output_dir: str):
        """
        Compiles and trains a random forest model for image classification.

        Args:
            output_dir (str): The directory to save the output files.

        Returns:
            pandas.DataFrame: The results of the model training and evaluation.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        all_paired_image_paths = pair_images_by_filename(self.rgb_dirs, self.disease_segmented_dirs, self.leaf_segmented_dirs)
        results = []

        combined_images, disease_masks, disease_types = self.load_and_preprocess_images(all_paired_image_paths)
        
        # Split the data into training and validation sets
        combined_images_train, combined_images_val, disease_masks_train, disease_masks_val, disease_types_train, disease_types_val = train_test_split(combined_images, disease_masks, disease_types, test_size=self.val_split, random_state=42)

        # Fit the scaler on the training data
        scaler = self.fit_scaler(combined_images_train)

        # Train the model on the entire training dataset
        rf_model = self.random_forest_classification(combined_images_train, disease_masks_train, scaler)
        labels_pred_train = rf_model.predict(scaler.transform(combined_images_train.reshape(-1, combined_images_train.shape[-1]))).reshape(-1, combined_images_train.shape[1])
        train_iou = calculate_iou(disease_masks_train, labels_pred_train)
        train_accuracy = accuracy_score(disease_masks_train.flatten(), labels_pred_train.flatten())

        # Predict on the validation set
        labels_pred_val = rf_model.predict(scaler.transform(combined_images_val.reshape(-1, combined_images_val.shape[-1]))).reshape(-1, combined_images_val.shape[1])
        val_iou = calculate_iou(disease_masks_val, labels_pred_val)
        val_accuracy = accuracy_score(disease_masks_val.flatten(), labels_pred_val.flatten())

        results.append({
            'Disease Type': 'All_Classes',
            'Train IoU': train_iou,
            'Validation IoU': val_iou,
            'Train Accuracy': train_accuracy,
            'Validation Accuracy': val_accuracy
        })

        # Evaluate performance for each disease type
        for disease in set(disease_types_val):
            # Filter the validation set for the current disease type
            idx = np.where(np.array(disease_types_val) == disease)[0]
            disease_masks_specific = disease_masks_val[idx]
            labels_pred_specific = labels_pred_val[idx]

            specific_iou = calculate_iou(disease_masks_specific, labels_pred_specific)
            specific_accuracy = accuracy_score(disease_masks_specific.flatten(), labels_pred_specific.flatten())

            results.append({
                'Disease Type': disease,
                'Validation IoU': specific_iou,
                'Validation Accuracy': specific_accuracy
            })

        # Create the results DataFrame and save to Excel
        df_all = pd.DataFrame(results)
        df_all.to_excel(os.path.join(output_dir, f"{self.dataset_name}_metrics.xlsx"), index=False)

        return df_all





