import os
import json
import mlflow
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

import utils as automl_utils
from models.ensemble import EnsembleModel
from models.stacking import StackingModel
from preprocessor import AutoMLPreprocessor

from mlops_client import MlopsClient
from polycloud_utils import PolyCloudStorageSupport

MLOPS_HOST = os.getenv('MLOPS_HOST')
MLFLOW_HOST = os.getenv('MLFLOW_HOST')
mlops = MlopsClient(MLOPS_HOST)
polycloud = PolyCloudStorageSupport()

class AutoMLModelTrain(BaseEstimator):
    
    def __init__(self, configs):
        self.configs = configs

        self.include_features = configs.get('include_features', 'all')
        self.validation_split_size = configs.get('validation_split_size', 0.2)
        self.task = configs.get('task', 'Classification')
        self.stacking = configs.get('stacking', False)
        self.ensemble = configs.get('ensemble', False)
        self.tune = configs.get('tune', False)
        self.include_models = configs.get('include_models', [])
        self.focus = configs.get('focus', None)

        self.experiment_name = configs.get('experiment_name', self.task + '_experiment')
        self.model_name = configs.get('model_name', None)

        # Some default models are set for training if not specified
        if not len(self.include_models):
            if self.task == 'Classification':
                self.include_models = ['LogisticRegression', 'XGBoost', 'DecisionTree']
            else:
                self.include_models = ['LinearRegression', 'XGBoost']

        # Some default metric is set for focus if not specified
        if not self.focus: 
            self.focus = 'accuracy' if self.task == 'Classification' else 'r2'

    def feature_selection(self, X_train, y_train, X_val):
        xgb = XGBClassifier() if self.task == 'Classification' else XGBRegressor()
        X_train_df = pd.DataFrame(X_train, columns = self.features)
        y_train_df = pd.DataFrame(y_train, columns = [y_train.name])
        X_val_df = pd.DataFrame(X_val, columns = self.features)
        xgb.fit(X_train_df, y_train_df)

        rfe = RFE(xgb, n_features_to_select=self.include_features, step=30)
        rfe.fit(X_train_df, y_train_df)
        X_train = rfe.transform(X_train_df)
        X_val = rfe.transform(X_val_df)

        selected_features = [feature for feature, status in zip(self.features, rfe.support_) if status]
        return X_train, X_val, selected_features

    def fit(self, X, y):
        
        # 1. Data Preprocessing
        preprocessor = AutoMLPreprocessor(self.configs)
        pipeline, X = preprocessor.preprocess_data(X)

        # 2. split the data into training and validation
        stratify = y if self.task == 'Classification' else None
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.validation_split_size,
                                                          stratify=stratify,
                                                          random_state=42)

        # 3. Do basic checks to ensure that there are no nulls values in the data and all the columns are numerical
        automl_utils.check_data(X_train, X_val)

        # 4. feature selection

        self.features = list(X_train.columns)
        if self.include_features != 'all':
            X_train, X_val, self.selected_features = self.feature_selection(X_train, y_train, X_val)
            
        else:
            self.selected_features = list(self.features)

        features_dict = {
            'all_features': self.features,
            'selected_features': self.selected_features
        }

        autotrain_outputs = []

        # 5. Model training

        if self.ensemble:
            
            self.ensemble_model = EnsembleModel(self.configs)
            self.ensemble_model.fit(X_train, y_train)

            model_identifier = automl_utils.generate_uuid()
            
            run_name = f"automl-ensemble-{model_identifier}"
            model_description = f"Creating an Ensemble of {self.include_models}"

            training_prediction = self.ensemble_model.predict(X_train)
            training_metrics = automl_utils.calculate_metrics(training_prediction, y_train, self.task)
            
            validation_prediction = self.ensemble_model.predict(X_val)
            validation_metrics = automl_utils.calculate_metrics(validation_prediction, y_val, self.task)

            features, scores = self.ensemble_model.interpret(X_train, X_val, y_val, self.features)

            try: 
                model_args = self.ensemble_model.model.get_params()
                model_args = {key: str(value) for key, value in model_args.items()}
            except: 
                model_args = {}

            serialized_model = automl_utils.serialize_model(self.ensemble_model.model)
            serialized_pipeline = automl_utils.serialize_model(pipeline)
            serialized_interpretation_plot = automl_utils.serialize_interpretation(features, scores, 'plot')
            serialized_interpretation_table = automl_utils.serialize_interpretation(features, scores, 'table')

            payload_contruction_dict = {
                'run_name': run_name,
                'experiment_name': self.experiment_name,
                'model_description': model_description,
                'task': self.task,
                'metrics' : {
                    'training_metrics': training_metrics,
                    'validation_metrics': validation_metrics
                },
                'model': serialized_model,
                'model_args': model_args,
                'model_architecture': 'ensemble',
                'model_library': automl_utils.architecture_library_mapping['ensemble'],
                'pipeline': serialized_pipeline,
                'visual_representation': serialized_interpretation_plot,
                'tabular_representation': serialized_interpretation_table
            }

            model_payload = automl_utils.construct_mlops_payload_for_model(payload_contruction_dict)
            autotrain_outputs.append([model_payload, self.configs, features_dict])

        elif self.stacking:

            self.stacking_model = StackingModel(self.configs)
            self.stacking_model.fit(X_train, y_train)

            model_identifier = automl_utils.generate_uuid()

            run_name = f"automl-stacking-{model_identifier}"
            model_description = f"Creating an Stacking of {self.include_models}"

            training_prediction = self.stacking_model.predict(X_train)
            training_metrics = automl_utils.calculate_metrics(training_prediction, y_train, self.task)
            
            validation_prediction = self.stacking_model.predict(X_val)
            validation_metrics = automl_utils.calculate_metrics(validation_prediction, y_val, self.task)

            features, scores = self.stacking_model.interpret(X_train, X_val, y_val, self.features)

            try: 
                model_args = self.stacking_model.model.get_params()
                model_args = {key: str(value) for key, value in model_args.items()}
            except: 
                model_args = {}

            serialized_model = automl_utils.serialize_model(self.stacking_model.model)
            serialized_pipeline = automl_utils.serialize_model(pipeline)
            serialized_interpretation_plot = automl_utils.serialize_interpretation(features, scores, 'plot')
            serialized_interpretation_table = automl_utils.serialize_interpretation(features, scores, 'table')

            payload_contruction_dict = {
                'run_name': run_name,
                'experiment_name': self.experiment_name,
                'model_description': model_description,
                'task': self.task,
                'metrics' : {
                    'training_metrics': training_metrics,
                    'validation_metrics': validation_metrics
                },
                'model': serialized_model,
                'model_args': model_args,
                'model_architecture': 'stacking',
                'model_library': automl_utils.architecture_library_mapping['stacking'],
                'pipeline': serialized_pipeline,
                'visual_representation': serialized_interpretation_plot,
                'tabular_representation': serialized_interpretation_table
            }

            model_payload = automl_utils.construct_mlops_payload_for_model(payload_contruction_dict)
            autotrain_outputs.append([model_payload, self.configs, features_dict])
  
        else:   
            # Neither ensemble nor stacking - train all the models seperately
            self.train_models = [automl_utils.models[self.task][model](self.configs) for model in self.include_models]

            for model, name in zip(self.train_models, self.include_models):

                if self.tune:
                    model.tune_and_fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)

                model_identifier = automl_utils.generate_uuid()

                run_name = f"automl-{name}-{model_identifier}"
                model_description = f"Training {name} model"

                training_prediction = model.predict(X_train)
                training_metrics = automl_utils.calculate_metrics(training_prediction, y_train, self.task)
                
                validation_prediction = model.predict(X_val)
                validation_metrics = automl_utils.calculate_metrics(validation_prediction, y_val, self.task)

                features, scores = model.interpret(X_train, X_val, y_val, self.features)

                try: 
                    model_args = model.model.get_params()
                    model_args = {key: str(value) for key, value in model_args.items()}
                except: 
                    model_args = {}

                serialized_model = automl_utils.serialize_model(model.model)
                serialized_pipeline = automl_utils.serialize_model(pipeline)
                serialized_interpretation_plot = automl_utils.serialize_interpretation(features, scores, 'plot')
                serialized_interpretation_table = automl_utils.serialize_interpretation(features, scores, 'table')

                payload_contruction_dict = {
                    'run_name': run_name,
                    'experiment_name': self.experiment_name,
                    'model_description': model_description,
                    'task': self.task,
                    'metrics' : {
                        'training_metrics': training_metrics,
                        'validation_metrics': validation_metrics
                    },
                    'model': serialized_model,
                    'model_args': model_args,
                    'model_architecture': name,
                    'model_library': automl_utils.architecture_library_mapping[name],
                    'pipeline': serialized_pipeline,
                    'visual_representation': serialized_interpretation_plot,
                    'tabular_representation': serialized_interpretation_table
                }

                model_payload = automl_utils.construct_mlops_payload_for_model(payload_contruction_dict)
                autotrain_outputs.append([model_payload, self.configs, features_dict])
        
        # 6. Mlops registration, save the training config and model features (used during inference)

        print('Model training done sucessfully')

        for model_output in autotrain_outputs:
            model_payload = model_output[0]
            training_config = model_output[1]
            model_features = model_output[2]

            # mlops registration
            model_response = mlops.set_config('models', model_payload)

            if not model_response:
                raise ValueError('Model registration in mlflow failed')
            
            experiment_id = model_response['data']['ml_client_model_config']['experiment_id']
            run_id = model_response['data']['ml_client_model_config']['run_id']
            
            print(f'Mlflow link {MLFLOW_HOST}/#/experiments/{experiment_id}/runs/{run_id}')
            
            # save the training config
            automl_utils.save_json_to_artifact_path(training_config, model_response, 'training_config')

            # save the model features
            automl_utils.save_json_to_artifact_path(model_features, model_response, 'model_features')
        
        return

    def predict(self, X):

        # 1.  get the model id using experiment name and model name
        model_id = automl_utils.get_model_id(self.experiment_name, self.model_name)
        
        # 2. get model configs using model_id
        model_config = mlops.get_config('models', model_id)

        # 3. get the model library and the artifact path
        artifact_path = model_config['data']['ml_client_model_config']['artifact_uri']
        key = '/'.join(artifact_path.split('/')[3:8])
        model_library = model_config['data']['model_parameters']['library']

        # 4. read the model features
        feature_list_path = key + '/model_features.json'
        features_dict_bytes = polycloud.read_file_from_cloud(feature_list_path)
        features_dict = json.load(features_dict_bytes)
        selected_features = features_dict['selected_features']

        # 5. read the training config
        training_config_path = key + '/training_config.json'
        training_config_bytes = polycloud.read_file_from_cloud(training_config_path)
        training_config = json.load(training_config_bytes)

        # 6. load the pipeline and the model that mlflow has logged
        logged_pipeline_path = f"{artifact_path}/pipeline"
        logged_model_path = f"{artifact_path}/model"

        pipeline = mlflow.sklearn.load_model(logged_pipeline_path)
        model =  automl_utils.mlflow_library_mapping[model_library].load_model(logged_model_path)

        # 7. apply the preprocess pipeline on the data and do the prediction
        preprocessor = AutoMLPreprocessor(training_config)
        X = preprocessor.apply_pipeline(X, pipeline, fit = False)

        print('preprocessed inference data')
        print(X.head())

        X = X[selected_features]
        return model.predict(X)
    
    def predict_proba(self, X):            
        
        # 1. get the model id using experiment name and model name
        model_id = automl_utils.get_model_id(self.experiment_name, self.model_name)
        
        # 2. get model configs using model_id
        model_config = mlops.get_config('models', model_id)

        # 3. get the model library and the artifact path
        artifact_path = model_config['data']['ml_client_model_config']['artifact_uri']
        key = '/'.join(artifact_path.split('/')[3:8])
        model_library = model_config['data']['model_parameters']['library']

        # 4. read the model features
        feature_list_path = key + '/model_features.json'
        features_dict_bytes = polycloud.read_file_from_cloud(feature_list_path)
        features_dict = json.load(features_dict_bytes)
        selected_features = features_dict['selected_features']

        # 5. read the training config
        training_config_path = key + '/training_config.json'
        training_config_bytes = polycloud.read_file_from_cloud(training_config_path)
        training_config = json.load(training_config_bytes)

        # 6. load the pipeline and the model that mlflow has logged
        logged_pipeline_path = f"{artifact_path}/pipeline"
        logged_model_path = f"{artifact_path}/model"

        pipeline = mlflow.sklearn.load_model(logged_pipeline_path)
        model =  automl_utils.mlflow_library_mapping[model_library].load_model(logged_model_path)

        # 7. apply the preprocess pipeline on the data and do the prediction
        preprocessor = AutoMLPreprocessor(training_config)
        X = preprocessor.apply_pipeline(X, pipeline, fit = False)

        X = X[selected_features]
        return model.predict_proba(X)