import optuna
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_score

from models.baseline import Classifier, focus_cv_mapping, Regressor

class KNNClassifier(Classifier):

    def __init__(self, params):
        super().__init__(params)
        self.model = KNeighborsClassifier()

    def tune_and_fit(self, X, y, fit = True, n_trials = 2):
        scoring = focus_cv_mapping[self.params['focus']]

        def objective(trial):
            # define the parameters
            param = {
                'n_neighbors': trial.suggest_int('n_neighbors', 5, 15),
                'p': trial.suggest_int('p', 2, 5),
            }

            model = KNeighborsClassifier(**param)
            score = cross_val_score(model, X, y, scoring=scoring, cv=self.cv_folds).mean()
            return score

        # the direction should be based on the metric
        if self.params['focus'] in ['accuracy', 'precision', 'recall', 'f1', 'r2', 'explained_variance']:
            direction = 'maximize'
        else:
            direction = 'minimize'
        
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        self.model = KNeighborsClassifier(**study.best_params)

        if fit:
            self.fit(X, y)

        return
    
class KNNRegressor(Regressor):

    def __init__(self, params):
        super().__init__(params)
        self.model = KNeighborsRegressor()

    def tune_and_fit(self, X, y, fit = True, n_trials = 2):
        scoring = focus_cv_mapping[self.params['focus']]

        def objective(trial):
            # define the parameters
            param = {
                'n_neighbors': trial.suggest_int('n_neighbors', 5, 15),
                'p': trial.suggest_int('p', 2, 5),
            }

            model = KNeighborsRegressor(**param)
            score = cross_val_score(model, X, y, scoring=scoring, cv=5).mean()
            return score

        # the direction should be based on the metric
        if self.params['focus'] in ['accuracy', 'precision', 'recall', 'f1', 'r2', 'explained_variance']:
            direction = 'maximize'
        else:
            direction = 'minimize'
        
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        self.model = KNeighborsRegressor(**study.best_params)

        if fit:
            self.fit(X, y)

        return