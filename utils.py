import io
import os
import uuid
import json
import base64
import mlflow
import cloudpickle
import pandas as pd
import matplotlib.pyplot as plt

from models.linearRegressor import Linear
from models.lassoRegressor import LassoRegression
from models.logisticRegression import Logistic
from models.naivebayesClassifier import NaiveBayes
from models.knn import KNNRegressor, KNNClassifier
from models.decisiontree import DecisiontreeClassifier, DecisiontreeRegressor
from models.randomforest import RandomforestClassifier, RandomforestRegressor
from models.xgboost import XGBoostClassifier, XGBoostRegressor
from metrics import ClassificationMetrics, RegressionMetrics

from mlops_client import MlopsClient
from polycloud_utils import PolyCloudStorageSupport

MLOPS_HOST = os.getenv('MLOPS_HOST')
mlops = MlopsClient(MLOPS_HOST)
polycloud = PolyCloudStorageSupport()

models = {
    'Classification': {
        'LogisticRegression': Logistic,
        'KNN': KNNClassifier,
        'NaiveBayesClassifier': NaiveBayes,
        'DecisionTree': DecisiontreeClassifier,
        'RandomForest': RandomforestClassifier,
        'XGBoost': XGBoostClassifier
    },

    'Regression': {
        'LinearRegression': Linear,
        'KNN': KNNRegressor,
        'LassoRegression': LassoRegression,
        'DecisionTree': DecisiontreeRegressor,
        'RandomForest': RandomforestRegressor,
        'XGBoost': XGBoostRegressor
    }
}

metrics = { 
    'Classification': ['accuracy',
                    'precision', 
                    'recall', 
                    'f1'
                    ],
    'Regression': ['r2', 
                   'mean_absolute_error',
                   'mean_squared_error',
                   'root_mean_squared_error',
                   'explained_variance',
                   'mean_absolute_percentage_error'
                   ]
}

architecture_library_mapping = {
    'LogisticRegression': 'scikit-learn',
    'KNN': 'scikit-learn',
    'NaiveBayesClassifier': 'scikit-learn',
    'XGBoost': 'xgboost',
    'DecisionTree': 'scikit-learn',
    'RandomForest': 'scikit-learn',
    'LinearRegression': 'scikit-learn',
    'LassoRegression': 'scikit-learn',
    'ensemble': 'scikit-learn',
    'stacking': 'scikit-learn'
}

mlflow_library_mapping = {
    "scikit-learn": mlflow.sklearn,
    "tensorflow": mlflow.tensorflow,
    "pytorch": mlflow.pytorch,
    "xgboost": mlflow.xgboost,
    "lightgbm": mlflow.lightgbm,
    "statsmodels": mlflow.statsmodels
}

def construct_mlops_payload_for_model(payload_contruction_dict): 

    model_payload = {
        'model_name': payload_contruction_dict['run_name'],
        'model_description': payload_contruction_dict['model_description'],
        'experiment_name': payload_contruction_dict['experiment_name'],
        'task': payload_contruction_dict['task'],
        'is_automl': True,

        'model_parameters': {
            'model_architecture': payload_contruction_dict['model_architecture'],
            'library': payload_contruction_dict['model_library'],
            'model_args': payload_contruction_dict['model_args']
        },

        'metrics': {
            'training_metrics': payload_contruction_dict['metrics']['training_metrics'],
            'validation_metrics': payload_contruction_dict['metrics']['validation_metrics']
        },

        'artifact_config': {
            'model_object': payload_contruction_dict['model'],
            'data_preprocessing_pipeline': [{'step_name': 'pipeline', 'preproc_object': payload_contruction_dict['pipeline']}],
        },

        'model_interpretability': {
            'feature_scores': {
                'visual_representation': payload_contruction_dict['visual_representation'],
                'tabular_representation': payload_contruction_dict['tabular_representation']
            }
        }
    }

    return model_payload

def check_data(X_train, X_val):
    X_train_null_values = X_train.isnull().any().any()
    X_train_categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns

    X_val_null_values = X_val.isnull().any().any()
    X_val_categorical_columns = X_val.select_dtypes(include=['object', 'category']).columns

    if X_train_null_values or len(X_train_categorical_columns):
        raise ValueError('The training dataset contains null values or categorical columns')
    
    if X_val_null_values or len(X_val_categorical_columns):
        raise ValueError('The validation dataset contains null values or categorical columns')

def calculate_metrics(prediction, y_test, task):
    metrics = ClassificationMetrics(y_test, prediction) if task == 'Classification' else RegressionMetrics(y_test, prediction)
    return metrics.evaluate_metrics()

def check_config(config, train):

    if train:
        task = config.get('task', None)
        ensemble = config.get('ensemble')
        stacking = config.get('stacking')
        focus = config.get('focus')
        experiment_name = config.get('experiment_name', None)

        if not task:
            raise NameError('Task must be specified')
        
        if not experiment_name:
            raise NameError('Experiment name must be specified')
        
        if task not in ['Classification', 'Regression']:
            raise NameError(f'The specified task {task} is invalid')
        
        if ensemble and stacking:
            raise ValueError('You cannot run both ensembling and stacking at the same time')

        if task == 'Classification' and focus and focus not in metrics['Classification']:
            raise NameError(f'The specified focus {focus} is invalid')
        
        if task == 'Regression' and focus and focus not in metrics['Regression']:
            raise NameError(f'The specified focus {focus} is invalid')

        for model in config.get('include_models', []):
            if model not in models[task]:
                raise NameError(f'The given model {model} does not exist')
            
    else:
        experiment_name = config.get('experiment_name', None)
        model_name = config.get('model_name', None)

        if not experiment_name:
            raise NameError('Experiment name must be specified')
        
        if not model_name:
            raise NameError('Model name must be specified')

def generate_uuid():
    return str(uuid.uuid4())

def create_dataframe_table(features, scores):
    return pd.DataFrame({'Feature': features, 'Score': scores})

def create_plot(features, scores):
    plt.figure(figsize=(10, 6))
    plt.barh(features, scores, color='skyblue')    
    plt.title('Feature Importance')
    plt.xlabel('Scores')
    plt.ylabel('Features')

def serialize_model(model):
    model_bytes = cloudpickle.dumps(model)
    model_base64 = base64.b64encode(model_bytes).decode('utf-8')
    return model_base64

def serialize_interpretation(features, scores, type = 'plot'):

    if type == 'plot':
        create_plot(features, scores)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        encoded_interpretation = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()
    
    else:
        buffer = io.BytesIO()
        table = create_dataframe_table(features, scores)
        table.to_csv(buffer, index=False)
        buffer.seek(0)
        encoded_interpretation = base64.b64encode(buffer.read()).decode('utf-8')
        buffer.close()

    return encoded_interpretation

def save_json_to_artifact_path(config, model_response, name):
    """
    Upload custom json to the mlflow artifact path. Like training config, feature list, etc.. that you want to track along with the mlflow artifact
    """
    
    path = model_response['data']['ml_client_model_config']['artifact_uri']
    key = '/'.join(path.split('/')[3:8])
    
    config = json.dumps(config).encode('utf-8')
    polycloud.upload_file_to_cloud(config, key + f'/{name}.json')

def get_model_id(experiment_name, model_name):
    params = {
        'search': experiment_name,
        'fields': 'experiment_name',
    }

    all_models_response = mlops.list_config('models', params)

    model_id = None
    for model in all_models_response['data']['results']:
        if model['model_name'] == model_name:
            model_id = model['model_id']
            break

    return model_id