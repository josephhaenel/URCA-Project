import numpy as np

def calculate_iou(self, y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    if np.sum(union) == 0:
        return 1.0 if np.sum(intersection) == 0 else 0  # Perfect IoU if true negatives
    else:
        iou_score = np.sum(intersection) / np.sum(union)
    return iou_score