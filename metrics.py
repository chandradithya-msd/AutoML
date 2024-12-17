import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)

class ClassificationMetrics:
    """
    Includes all the common metrics used for classification tasks
    Can also include some custom metrics based on your use case

    """

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
    
    def precision(self):
        return precision_score(self.y_true, self.y_pred)

    def recall(self):
        return recall_score(self.y_true, self.y_pred)

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def f1_score(self):
        return f1_score(self.y_true, self.y_pred)

    def specificity(self):
        tn, fp, _, _ = confusion_matrix(self.y_true, self.y_pred).ravel()
        return tn / (tn + fp)
    
    def evaluate_metrics(self):
        metrics = {}
        metrics['accuracy'] = self.accuracy()
        metrics['precision'] = self.precision()
        metrics['recall'] = self.recall()
        metrics['f1'] = self.f1_score()
        metrics['specificity'] = self.specificity()

        return metrics


class RegressionMetrics:
    """
    Includes all the common metrics used for regression tasks
    Can also include some custom metrics based on your use case
    """
    
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def mse(self):
        return mean_squared_error(self.y_true, self.y_pred)

    def mae(self):
        return mean_absolute_error(self.y_true, self.y_pred)

    def rmse(self):
        return np.sqrt(mean_squared_error(self.y_true, self.y_pred))

    def r2_score(self):
        return r2_score(self.y_true, self.y_pred)

    def explained_variance(self):
        return explained_variance_score(self.y_true, self.y_pred)
    
    def evaluate_metrics(self):
        metrics = {}
        metrics['r2'] = self.r2_score()
        metrics['mean_absolute_error'] = self.mae()
        metrics['mean_squared_error'] = self.mse()
        metrics['root_mean_squared_error'] = self.rmse()
        metrics['explained_variance'] = self.explained_variance()
        
        return metrics