import shap
import numpy as np
import warnings
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

focus_cv_mapping = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'r2': 'r2',
    'mean_absolute_error': 'neg_mean_absolute_error',
    'mean_squared_error': 'neg_mean_squared_error',
    'root_mean_squared_error': 'neg_root_mean_squared_error',
    'explained_variance': 'explained_variance',
    'mean_absolute_percentage_error': 'neg_mean_absolute_percentage_error'
}

class Regressor(BaseEstimator, RegressorMixin):
    """
    This is the base class for all regression models.
    """

    def __init__(self, params):
        self.params = params
        self.model = None
        self.model_file_path = params.get('model_file_path')
        self.cv_folds = params.get("cv_folds")
    
    def fit(self, X, y):
        self.model.fit(X,y)

    def interpret(self, X_train, X_test, y_test, features):
        average_shap_values_dict = None
        try: 
            explainer = shap.Explainer(self.model, X_train)
            shap_values = explainer.shap_values(X_test)
            average_shap_values = np.mean(np.abs(shap_values), axis=0)
            feature_scores = [[score, feature] for feature, score in zip(features, average_shap_values)]
            feature_scores.sort(reverse = True)                
            average_shap_values_dict = {name: value for value, name in feature_scores[:10]}

            top_features = average_shap_values_dict.keys()
            scores = average_shap_values_dict.values()
            scores = [score / sum(scores) for score in scores]
            return top_features, scores
        
        except:
            importance_scores = permutation_importance(self.model, X_test, y_test, n_repeats=10, random_state=42).importances_mean
            feature_scores = [[score, feature] for score, feature in zip(importance_scores, features)]
            feature_scores.sort(reverse = True)
            top_features = [feature[1] for feature in feature_scores[:10]]
            scores = [feature[0] for feature in feature_scores[:10]]
            scores = [score / sum(scores) for score in scores]
            return top_features, scores
    
    def predict(self, X):
        return self.model.predict(X)
    

class Classifier(BaseEstimator, ClassifierMixin):
    """
    This is the base class for all classification models.
    """

    def __init__(self, params):
        self.params = params
        self.model = None
        self.cv_folds = params.get("cv_folds")
    
    def fit( self, X, y):
        self.model.fit(X,y)
        self.classes_ = self.model.classes_

    def interpret(self, X_train, X_test, y_test, features):
        average_shap_values_dict = None
        try:
            explainer = shap.Explainer(self.model, X_train)
            shap_values = explainer.shap_values(X_test)
            average_shap_values = np.mean(np.abs(shap_values), axis=0)
            feature_scores = [[score, feature] for feature, score in zip(features, average_shap_values)]
            feature_scores.sort(reverse = True)                
            average_shap_values_dict = {name: value for value, name in feature_scores[:10]}

            top_features = average_shap_values_dict.keys()
            scores = average_shap_values_dict.values()
            scores = [score / sum(scores) for score in scores]
            return top_features, scores

        except:
            importance_scores = permutation_importance(self.model, X_test, y_test, n_repeats=10, random_state=42).importances_mean
            feature_scores = [[score, feature] for score, feature in zip(importance_scores, features)]
            feature_scores.sort(reverse = True)
            top_features = [feature[1] for feature in feature_scores[:10]]
            scores = [feature[0] for feature in feature_scores[:10]]
            scores = [score / sum(scores) for score in scores]
            return top_features, scores
    
    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self,X):
        y = self.model.predict(X)
        if "num_classes" in self.params and self.params["num_classes"]:
            return y
        return np.column_stack((y-1,y)) # p-1, p at row level