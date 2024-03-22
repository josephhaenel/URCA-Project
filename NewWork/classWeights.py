import numpy as np
from collections import defaultdict
from PIL import Image
from keras import backend as K

class ClassWeightManager:
    def __init__(self, class_mapping):
        self.class_mapping = class_mapping
        self.class_weights = None

    def compute_class_weights(self, mask_paths):
        class_counts = defaultdict(int)

        # Count the pixels for each class in all masks
        for mask_path in mask_paths:
            mask = np.array(Image.open(mask_path).convert('L'))
            for value in self.class_mapping.values():
                class_counts[value] += np.sum(mask == value)

        total_pixels = sum(class_counts.values())

        # Compute the weights inversely proportional to class frequencies
        self.class_weights = {class_id: total_pixels / (len(self.class_mapping) * count)
                              for class_id, count in class_counts.items()}

        # Normalize the weights so that the smallest weight is 1.0
        min_weight = min(self.class_weights.values())
        self.class_weights = {class_id: weight / min_weight
                              for class_id, weight in self.class_weights.items()}

    def get_class_weights(self):
        if self.class_weights is None:
            raise ValueError("Class weights have not been computed yet.")
        return self.class_weights

    def get_weighted_loss(self):
        weights = np.array(list(self.get_class_weights().values()))

        def weighted_categorical_crossentropy(y_true, y_pred):
            Kweights = K.constant(weights)
            y_true = K.one_hot(K.cast(y_true, 'int32'), num_classes=len(weights))[..., 0]
            return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)

        return weighted_categorical_crossentropy
