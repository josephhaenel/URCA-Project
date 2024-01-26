import tensorflow as tf

class IoUMetric(tf.keras.metrics.Metric):
    def __init__(self, name='iou_score', **kwargs):
        super(IoUMetric, self).__init__(name=name, **kwargs)
        self.iou = self.add_weight(name='iou', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_float = tf.cast(y_true, tf.float32)
        y_pred_float = tf.cast(y_pred, tf.float32)
        y_pred_thresholded = tf.cast(tf.greater(y_pred_float, 0.5), tf.float32)

        intersection = tf.reduce_sum(y_true_float * y_pred_thresholded)
        union = tf.reduce_sum(y_true_float) + tf.reduce_sum(y_pred_thresholded) - intersection
        iou_score = intersection / (union + tf.keras.backend.epsilon())
        
        self.iou.assign(iou_score)

    def result(self):
        return self.iou

    def reset_states(self):
        self.iou.assign(0.0)
