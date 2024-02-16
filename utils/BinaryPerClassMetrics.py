import tensorflow as tf

class BinaryPerClassMetrics(tf.keras.metrics.Metric):
    def __init__(self, disease_types, threshold=0.5, name='binary_per_class_metrics', **kwargs):
        super(BinaryPerClassMetrics, self).__init__(name=name, **kwargs)
        self.disease_types = disease_types
        self.threshold = threshold
        self.unique_diseases = sorted(set(disease_types))
        self.num_diseases = len(self.unique_diseases)
        self.iou_scores = self.add_weight(name='iou_scores', initializer='zeros', shape=(self.num_diseases,))
        self.accuracy_scores = self.add_weight(name='accuracy_scores', initializer='zeros', shape=(self.num_diseases,))

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions and true values to binary format based on the threshold
        y_pred_thresholded = tf.cast(y_pred > self.threshold, tf.float32)
        for i, disease in enumerate(self.unique_diseases):
            # Filter by disease type
            mask = tf.cast(tf.equal(self.disease_types, disease), tf.float32)
            if tf.reduce_sum(mask) == 0:  # Skip if no samples for this disease
                continue

            # Calculate IoU and accuracy for the current disease type
            intersection = tf.reduce_sum(y_true * y_pred_thresholded * mask)
            union = tf.reduce_sum((y_true + y_pred_thresholded) * mask) - intersection
            iou_score = intersection / (union + tf.keras.backend.epsilon())

            correct_predictions = tf.equal(y_true, tf.round(y_pred_thresholded))
            accuracy_score = tf.reduce_sum(tf.cast(correct_predictions, tf.float32) * mask) / tf.reduce_sum(mask)

            # Update the scores
            self.iou_scores[i].assign_add(iou_score)
            self.accuracy_scores[i].assign_add(accuracy_score)

    def result(self):
        return {
            "iou_per_disease": self.iou_scores / tf.cast(tf.size(self.disease_types), tf.float32),
            "accuracy_per_disease": self.accuracy_scores / tf.cast(tf.size(self.disease_types), tf.float32),
        }

    def reset_states(self):
        self.iou_scores.assign(tf.zeros_like(self.iou_scores))
        self.accuracy_scores.assign(tf.zeros_like(self.accuracy_scores))
