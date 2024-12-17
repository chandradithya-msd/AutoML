import shap
import numpy as np
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.inspection import permutation_importance

from utils import models

class StackingModel:
    def __init__(self, params):
        self.params = params
        self.task = params.get('task')
        self.tune = params.get('tune')
        self.save_artifacts = params.get('save_artifacts')
        self.include_models = params.get('include_models')
        self.model = None
    
    def fit(self, X, y):
        estimators = [(name, models[self.task][name](self.params)) for name in self.include_models]
        if self.tune:
            for _, model in estimators:
                model.tune_and_fit(X, y, fit = False)

        base_estimators = [(name, model.model) for name, model in estimators]
        
        if self.task == 'Classification':
            self.model = StackingClassifier(estimators = base_estimators)
        elif self.task == 'Regression':
            self.model = StackingRegressor(estimators = base_estimators)

        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if self.task == 'Regression':
            raise TypeError('Predict proba is not applicable for regression tasks')
        
        return self.model.predict_proba(X)
    
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
