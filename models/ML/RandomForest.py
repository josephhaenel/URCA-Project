import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
from keras.preprocessing.image import load_img, img_to_array
from ML.MachineLearningUtils.PairImagesByFilename import pair_images_by_filename
from ML.MachineLearningUtils.calculate_iou import calculate_iou

class ImageRandomForestSegmenter:
    def __init__(self, rgb_dirs: list[str], disease_segmented_dirs: list[str], leaf_segmented_dirs: list[str], val_split: float, dataset_name: str) -> None:
        self.rgb_dirs = rgb_dirs
        self.disease_segmented_dirs = disease_segmented_dirs
        self.leaf_segmented_dirs = leaf_segmented_dirs
        self.val_split = val_split
        self.dataset_name = dataset_name
        
    def evaluate_on_full_dataset(self, combined_images, disease_masks):
        scaler = self.fit_scaler(combined_images)
        labels_pred = self.random_forest_classification(combined_images, scaler)
        full_iou = self.calculate_iou(disease_masks, labels_pred)
        full_accuracy = accuracy_score(disease_masks.flatten(), labels_pred.flatten())
        return full_iou, full_accuracy
    
    def random_forest_classification(self, combined_images, scaler):
        # Apply the provided scaler to standardize the data
        combined_images_standardized = scaler.transform(combined_images.reshape(-1, combined_images.shape[-1]))
        rf = RandomForestClassifier(random_state=42)
        labels = rf.fit_predict(combined_images_standardized)
        return labels.reshape(-1, combined_images.shape[1])
    



