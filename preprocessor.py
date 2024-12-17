import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import ( 
    PowerTransformer, 
    RobustScaler, 
    OrdinalEncoder, 
    OneHotEncoder, 
    StandardScaler, 
    MinMaxScaler,
)

class MathTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, operation='square'):
        self.operation = operation
        self.operations_map = {
            'square': np.square,
            'cube': lambda x: np.power(x, 3),
            'log': np.log,
            'absolute': np.abs,
            'inverse': lambda x: 1 / x,
            'exponential': np.exp,
            'reciprocal': lambda x: 1 / np.maximum(x, 1e-10),
            'square_root': np.sqrt,
            'cube_root': np.cbrt,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        func = self.operations_map.get(self.operation)

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(func(X.values), columns=X.columns, index=X.index)
        else:
            return func(X)

    def get_feature_names_out(self, input_features=None):
        return input_features

step_method_mapping = {
    'impute': {
        'mean': SimpleImputer(strategy='mean'),
        'median': SimpleImputer(strategy='median'),
        'mode': SimpleImputer(strategy='most_frequent'),
        'custom': SimpleImputer
    },
    'encode': {
        'label': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
        'one_hot': OneHotEncoder()
    },
    'scale': {
        'standard': StandardScaler(),
        'min_max': MinMaxScaler(),
    },
    'skew': {
        'log': MathTransformer('log'),
        'cube_root': MathTransformer('cube_root'),
        'square_root': MathTransformer('square_root'),
        'reciprocal': MathTransformer('reciprocal'),
        'exponential': MathTransformer('exponential'),
        'inverse': MathTransformer('inverse'),
        'absolute': MathTransformer('absolute'),
        'square': MathTransformer('square'),
        'cube': MathTransformer('cube'),
        'yeo_johnson': PowerTransformer(method='yeo-johnson'),
        'box_cox': PowerTransformer(method='box-cox')
    },
    'outlier': {
        'robust': RobustScaler()
    },
    'drop': 'drop',
    'fit_numerical_to_categorical': 'fit_numerical_to_categorical'
}

class AutoMLPreprocessor:
    """
    This class is responsible for preprocessing the data based on the specified configuration.
    It creates a pipeline that defines the preprocessing steps that needs to be applied to the data.
    It supports the following preprocessing steps:
        - Impute
        - Scale
        - Encode
        - Skew handle
        - Outlier handle
    """

    def __init__(self, preproc_config):
        self.ignore_columns_for_training = preproc_config.get('ignore_columns_for_training', [])
        self.fit_numerical_to_categorical = preproc_config.get('fit_numerical_to_categorical', [])
        self.preproc_steps = preproc_config.get('preproc_steps', [])

        self.step_method_mapping = step_method_mapping

    def initial_preprocessing(self, data):
        data = data[[col for col in data.columns if col not in self.ignore_columns_for_training]]
        for col in self.fit_numerical_to_categorical:
            data[col] = data[col].astype(str)
        
        return data

    def get_columns_to_use(self, pipeline):
        
        cols_to_use = []
        for _, _, columns in pipeline.transformers:
            cols_to_use.extend(columns)
        return cols_to_use
    
    def create_pipeline(self):
        """
        Keep track of the transformations required for each column
        ColumnTransformer: Applies the transformations parallely
        Pipeline : Applies the transformations sequentially

        The entire structure of the preprocessor would look like 
        ColumnTransformer((Pipeline(step1, step2, ...), col1),
                        (Pipeline(step1, step2, ...), col2),
                        (Pipeline(step1, step2, ...), col3), ....)

        The transformations for all the columns happens parallely whereas the transformations for a single column has some sequential steps

        """
        transformers = []
        column_pipeline_dict = defaultdict(list)

        for preproc_step in self.preproc_steps:
            step = preproc_step['step']
            method = preproc_step['method']
            columns_to_include = preproc_step['columns_to_include']
            value = preproc_step.get('value', 0) 
            
            transformation_function = self.step_method_mapping[step][method]
            if step == 'impute' and method == 'custom':
                transformation_function = transformation_function(strategy='constant', fill_value=value)

            for col in columns_to_include:
                step_count = len(column_pipeline_dict[col]) + 1
                column_pipeline_dict[col].append((f'Step {step_count} - {step}', transformation_function))

        for col, _ in column_pipeline_dict.items():
            column_pipeline_dict[col] = Pipeline(
                steps = column_pipeline_dict[col]
            )
        
        for col, pipeline in column_pipeline_dict.items():
            transformers.append((f'{col}_pipeline', pipeline, [col]))

        preprocessor = ColumnTransformer(transformers = transformers, remainder='passthrough')
        return preprocessor
    
    def apply_pipeline(self, data, pipeline, fit = True):
        
        data = self.initial_preprocessing(data)

        cols_to_use = self.get_columns_to_use(pipeline)

        # check if the columns required by the pipeline is present in the data
        for col in cols_to_use:
            if col not in data.columns:
                raise KeyError(f"Column '{col}' not found in the data.")

        required_data = data[cols_to_use] # columns on which preprocessing is required
        ignored_data = data.drop(cols_to_use, axis=1) # columns on which preprocessing is not required

        if fit:
            processed_data = pipeline.fit_transform(required_data)
        else:
            processed_data = pipeline.transform(required_data)
        
        output_columns = list(pipeline.get_feature_names_out())
        output_columns = [col.split('__')[1] for col in output_columns]
        processed_data = pd.DataFrame(processed_data, columns = output_columns)

        # concat to get final data
        final_data = pd.concat([processed_data, ignored_data], axis=1)

        # reorder the processed data in ascending order by column names
        # pipeline transform automatically reorders the data in some way. So reordering this again makes it easy to track the columns
        final_data = final_data.reindex(sorted(final_data.columns), axis=1)

        return final_data

    def preprocess_data(self, data):

        # create the pipeline
        pipeline = self.create_pipeline()

        # apply the pipeline to the data
        processed_data = self.apply_pipeline(data, pipeline)
        return pipeline, processed_data
