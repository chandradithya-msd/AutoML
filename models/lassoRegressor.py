import optuna
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

from models.baseline import Regressor, focus_cv_mapping

class LassoRegression(Regressor):

    def __init__(self, params):
        super().__init__(params)
        self.model = Lasso()

    def tune_and_fit(self, X, y, fit = True, n_trials = 2):
        scoring = focus_cv_mapping[self.params['focus']]

        def objective(trial):
            # define the parameters
            param = {
                'alpha': trial.suggest_categorical('alpha', [0.1, 0.01, 0.001, 0.2]),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                'positive': trial.suggest_categorical('positive', [True, False]),
                'selection': trial.suggest_categorical('selection', ['cyclic', 'random'])
            }

            model = Lasso(**param)
            score = cross_val_score(model, X, y, scoring=scoring, cv=self.cv_folds).mean()
            return score

        # the direction should be based on the metric
        if self.params['focus'] in ['accuracy', 'precision', 'recall', 'f1', 'r2', 'explained_variance']:
            direction = 'maximize'
        else:
            direction = 'minimize'
        
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        self.model = Lasso(**study.best_params)

        if fit:
            self.fit(X, y)

        return