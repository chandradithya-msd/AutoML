import optuna
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import cross_val_score

from models.baseline import Classifier, focus_cv_mapping, Regressor

class XGBoostClassifier(Classifier):

    def __init__(self, params):
        super().__init__(params)
        self.model = XGBClassifier()

    def tune_and_fit(self, X, y, fit = True, n_trials = 2):
        scoring = focus_cv_mapping[self.params['focus']]

        def objective(trial):
            # define the parameters
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'lambda': trial.suggest_float('lambda', 0, 5),
                'alpha': trial.suggest_float('alpha', 0, 5),
            }

            model = XGBClassifier(**params)
            score = cross_val_score(model, X, y, scoring=scoring, cv=self.cv_folds).mean()
            return score

        # the direction should be based on the metric
        if self.params['focus'] in ['accuracy', 'precision', 'recall', 'f1', 'r2', 'explained_variance']:
            direction = 'maximize'
        else:
            direction = 'minimize'
        
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        self.model = XGBClassifier(**study.best_params)

        if fit:
            self.fit(X, y)

        return
    
    def interpret(self, X_train, X_test, y_test, features):
        scores = self.model.feature_importances_
        feature_scores = [[score, feature] for score, feature in zip(scores, features)]
        feature_scores.sort(reverse = True)
        top_features = [feature[1] for feature in feature_scores[:10]]
        scores = [feature[0] for feature in feature_scores[:10]]
        scores = [score / sum(scores) for score in scores]
        return top_features, scores
    
class XGBoostRegressor(Regressor):

    def __init__(self, params):
        super().__init__(params)
        self.model = XGBRegressor()

    def tune_and_fit(self, X, y, fit=True, n_trials=2):
        scoring = focus_cv_mapping[self.params['focus']]

        def objective(trial):
            # Define the parameters to tune
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'lambda': trial.suggest_float('lambda', 0, 5),
                'alpha': trial.suggest_float('alpha', 0, 5)
            }

            model = XGBRegressor(**params)

            scores = cross_val_score(model, X, y, scoring=scoring, cv=5)
            mean_score = scores.mean()
            return mean_score

        # Set the direction based on the metric
        if self.params['focus'] in ['r2', 'explained_variance']:
            direction = 'maximize'
        else:
            direction = 'minimize'

        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)

        self.model = XGBRegressor(**study.best_params)

        if fit:
            self.fit(X, y)

        return
    
    def interpret(self, X_train, X_test, y_test, features):
        scores = self.model.feature_importances_
        feature_scores = [[score, feature] for score, feature in zip(scores, features)]
        feature_scores.sort(reverse = True)
        top_features = [feature[1] for feature in feature_scores[:10]]
        scores = [feature[0] for feature in feature_scores[:10]]
        scores = [score / sum(scores) for score in scores]
        return top_features, scores
