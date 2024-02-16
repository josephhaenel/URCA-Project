from utils.BinaryPerClassMetrics import BinaryPerClassMetrics
import tensorflow as tf
import os

class LogPerClassMetrics(tf.keras.callbacks.Callback):
    def __init__(self, metric: BinaryPerClassMetrics, output_dir: str, filename='average_per_class_metrics.txt'):
        super().__init__()
        self.metric = metric
        self.filepath = os.path.join(output_dir, filename)
        # Initialize storage for sums and counts to compute averages
        self.iou_sums = tf.Variable(tf.zeros(len(metric.class_names)), trainable=False)
        self.accuracy_sums = tf.Variable(tf.zeros(len(metric.class_names)), trainable=False)
        self.epoch_count = tf.Variable(0, trainable=False)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count.assign_add(1)
        metrics_result = self.metric.result()
        iou_per_class = metrics_result["iou_per_class"].numpy()
        accuracy_per_class = metrics_result["accuracy_per_class"].numpy()
        
        # Update running sums
        self.iou_sums.assign_add(iou_per_class)
        self.accuracy_sums.assign_add(accuracy_per_class)

        # Cast epoch_count to float for division
        epoch_count_float = tf.cast(self.epoch_count, tf.float32)

        epsilon = 1e-10  # Small constant to prevent division by zero

        # Compute running averages
        iou_averages = self.iou_sums / (epoch_count_float + epsilon)
        accuracy_averages = self.accuracy_sums / (epoch_count_float + epsilon)

        with open(self.filepath, 'a') as f:
            f.write(f'Epoch {epoch + 1} Averages\n')
            for class_id, class_name in enumerate(self.metric.class_names):
                iou_avg = iou_averages[class_id].numpy()  # Convert to numpy for easy formatting
                accuracy_avg = accuracy_averages[class_id].numpy()
                f.write(f'{class_name} - Average IoU: {iou_avg}, Average Accuracy: {accuracy_avg}\n')
            f.write('\n')


    def on_train_begin(self, logs=None):
        # Reset sums and count at the beginning of training
        self.iou_sums.assign(tf.zeros_like(self.iou_sums))
        self.accuracy_sums.assign(tf.zeros_like(self.accuracy_sums))
        self.epoch_count.assign(0)
