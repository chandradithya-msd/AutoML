import optuna
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

from models.baseline import Classifier, focus_cv_mapping

class NaiveBayes(Classifier):
    
    def __init__(self, params):
        super().__init__(params)
        self.model = GaussianNB()

    def tune_and_fit(self, X, y, fit = True, n_trials = 2):
        scoring = focus_cv_mapping[self.params['focus']]

        def objective(trial):
            # define the parameters
            param = {
                'var_smoothing': trial.suggest_categorical('var_smoothing', [1e-10, 1e-5, 1e-9, 1e-8])   
            }

            model = GaussianNB(**param)
            score = cross_val_score(model, X, y, scoring=scoring, cv=self.cv_folds).mean()
            return score

        # the direction should be based on the metric
        if self.params['focus'] in ['accuracy', 'precision', 'recall', 'f1', 'r2', 'explained_variance']:
            direction = 'maximize'
        else:
            direction = 'minimize'
        
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)
        self.model = GaussianNB(**study.best_params)

        if fit:
            self.fit(X, y)

        return