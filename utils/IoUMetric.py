import tensorflow as tf
import os

class IoUMetric(tf.keras.metrics.Metric):
    def __init__(self, name='IoU', **kwargs):
        super(IoUMetric, self).__init__(name=name, **kwargs)
        self.total_iou = self.add_weight(name='iou', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_float = tf.cast(y_true, tf.float32)
        y_pred_float = tf.cast(y_pred, tf.float32)
        y_pred_thresholded = tf.cast(tf.greater(y_pred_float, 0.5), tf.float32)

        intersection = tf.reduce_sum(y_true_float * y_pred_thresholded)
        union = tf.reduce_sum(y_true_float) + tf.reduce_sum(y_pred_thresholded) - intersection
        iou_score = intersection / (union + tf.keras.backend.epsilon())

        self.total_iou.assign_add(iou_score)
        self.count.assign_add(1)

    def result(self):
        return self.total_iou / self.count

    def reset_states(self):
        self.total_iou.assign(0.0)
        self.count.assign(0.0)
    
    @property
    def variables(self):
        return [self.total_iou, self.count]
        
class IoULogger(tf.keras.callbacks.Callback):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        with open(os.path.join(self.output_dir, 'iou_log.txt'), 'a') as file:
            file.write(f"Epoch {epoch + 1}\n")
            for key, value in logs.items():
                if key == 'iou_score':
                    file.write(f"{key}: {value}\n")


