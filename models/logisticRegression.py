import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from models.baseline import Classifier, focus_cv_mapping

class Logistic(Classifier):

    def __init__(self, params):
        super().__init__(params)
        self.model = LogisticRegression()

    def tune_and_fit(self, X, y, fit = True, n_trials = 2):
        scoring = focus_cv_mapping[self.params['focus']]

        def objective(trial):
            # define the parameters
            param = {
                'penalty': trial.suggest_categorical('penalty', ['l2', None]),
                'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
            }

            model = LogisticRegression(**param)
            score = cross_val_score(model, X, y, scoring=scoring, cv=self.cv_folds).mean()
            return score

        # the direction should be based on the metric
        if self.params['focus'] in ['accuracy', 'precision', 'recall', 'f1', 'r2', 'explained_variance']:
            direction = 'maximize'
        else:
            direction = 'minimize'
        
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        self.model = LogisticRegression(**study.best_params)

        if fit:
            self.fit(X, y)

        return