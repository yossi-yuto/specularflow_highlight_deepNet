import numpy as np
from sklearn.metrics import precision_recall_curve


def calc_maxFbeta(true_1d, pred_1d):
    assert len(true_1d.shape) == 1 and len(pred_1d.shape) == 1, f"No format input.shape is not 1 dimension."
    precisions, recalls, _ = precision_recall_curve(true_1d, pred_1d)
    numerator = (1 + 0.3) * (precisions * recalls)
    denom = (0.3 * precisions) + recalls
    fbetas = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
    return np.max(fbetas)