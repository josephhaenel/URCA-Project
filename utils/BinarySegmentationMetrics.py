import tensorflow as tf
import numpy as np
import os
import pandas as pd
from keras.callbacks import Callback

class BinarySegmentationMetrics(Callback):
    def __init__(self, validation_data, validation_disease_types, model_name, learning_rate, val_split, dataset_name, output_dir):
        super(BinarySegmentationMetrics, self).__init__()
        self.validation_data = validation_data
        self.validation_disease_types = validation_disease_types
        self.results = []  # Store results here
        # Additional attributes
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.val_split = val_split
        self.dataset_name = dataset_name
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.validation_data[0])
        binary_predictions = (predictions > 0.5).astype(int)
        unique_diseases = set(self.validation_disease_types)

        for disease in unique_diseases:
            indices = np.where(self.validation_disease_types == disease)[0]
            disease_specific_true_labels = self.validation_data[1][indices]
            disease_specific_predictions = binary_predictions[indices]

            # Calculate IoU and Accuracy
            iou_metric = tf.keras.metrics.MeanIoU(num_classes=2)
            iou_metric.update_state(disease_specific_true_labels, disease_specific_predictions)
            accuracy_metric = tf.keras.metrics.BinaryAccuracy()
            accuracy_metric.update_state(disease_specific_true_labels, disease_specific_predictions)
            # Accumulate results
            self.results.append({
                'Epoch': epoch + 1,
                'Disease Type': disease,
                'IoU': iou_metric.result().numpy(),
                'Accuracy': accuracy_metric.result().numpy(),
                'Model': self.model_name,
                'Learning Rate': self.learning_rate,
                'Validation Split': self.val_split,
                'Dataset Name': self.dataset_name
            })

    def save_results_to_excel(self, file_name='model_results.xlsx'):
        df = pd.DataFrame(self.results)
        os.makedirs(self.output_dir, exist_ok=True)  # This line will create the directory if it doesn't exist
        file_path = os.path.join(self.output_dir, file_name)
        mode = 'a' if os.path.exists(file_path) else 'w'
        if mode == 'a':
            with pd.ExcelWriter(file_path, engine='openpyxl', mode=mode, if_sheet_exists='overlay') as writer:
                df.to_excel(writer, sheet_name=f'{self.model_name}_{self.dataset_name}', index=False)
        else:
            with pd.ExcelWriter(file_path, engine='openpyxl', mode=mode) as writer:
                df.to_excel(writer, sheet_name=f'{self.model_name}_{self.dataset_name}', index=False)


