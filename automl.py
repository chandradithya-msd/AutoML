import utils as automl_utils
from model import AutoMLModelTrain

import warnings
warnings.filterwarnings('ignore')

class AutoML:

    def __init__(self, configs, train = True):

        self.configs = configs
        self.task = configs.get('task', None)
        self.autotrain = AutoMLModelTrain(self.configs)

        # check model config here - throws error if the configs are not correctly configured
        automl_utils.check_config(self.configs, train)

    def fit(self, X, y):
        """
        This function does the following:
        1. Validation split - will be used to see the model preformance
        2. Preprocessing
        3. Basic data checks
        4. Feature selection
        5. Model training
        6. Collects the mlops payload and does the model registration in mlflow
        7. Saves the model features, configs that will be used during inference
        """

        # AutoML AutoTrain + AutoML Preprocessor + MLops
        self.autotrain.fit(X, y)
        return
        
    def predict(self, X):
        """
        1. Loads the model and pipeline from the mlflow logged artifacts
        2. Does preprocessing on the inference data
        3. Makes predictions
        """

        return self.autotrain.predict(X)
    
    def predict_proba(self, X):
        """
        1. Loads the model and pipeline from the mlflow logged artifacts
        2. Does preprocessing on the inference data
        3. Computes prediction probabilities
        """

        if self.task == 'Regression':
            raise TypeError('Predict proba is not applicable for regression tasks')

        return self.autotrain.predict_proba(X)
