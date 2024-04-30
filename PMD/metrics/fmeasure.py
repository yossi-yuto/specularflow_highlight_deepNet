import numpy as np
from sklearn.metrics import precision_recall_curve

def calculate_max_fbeta(y_true, y_scores):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    beta_square = 0.3
    zero_div_values = np.zeros_like(precisions)
    numerator = (1 + beta_square) * precisions * recalls
    denominator = (beta_square * precisions) + recalls
    Fscores =  np.divide(numerator, denominator, out=zero_div_values, where=denominator!=0.)
    assert np.nan not in Fscores, "nan value in Fbeta score"
    return Fscores.max(), thresholds[Fscores.argmax()]
