import tensorflow as tf
from keras.metrics import MeanIoU

class MeanIoUMetric(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name='mean_iou', **kwargs):
        super(MeanIoUMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.mean_iou = MeanIoU(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        self.mean_iou.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.mean_iou.result()

    def reset_state(self):
        self.mean_iou.reset_state()

    def get_config(self):
        config = super(MeanIoUMetric, self).get_config()
        config['num_classes'] = self.num_classes
        return config

