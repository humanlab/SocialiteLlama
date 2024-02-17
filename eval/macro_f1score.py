import numpy as np
from sklearn.metrics import f1_score

def macro_f1_score(y_true, y_pred):
    """
    Calculate the macro-F1 score.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.

    Returns:
    float: Macro-F1 score.
    """
    unique_labels = np.unique(y_true)
    macro_f1 = 0

    for label in unique_labels:
        true_label = (y_true == label)
        pred_label = (y_pred == label)
        f1 = f1_score(true_label, pred_label)
        macro_f1 += f1

    macro_f1 /= len(unique_labels)
    return macro_f1