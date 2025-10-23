import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object

from src.exception import CustomException
from src.logger import logging



@dataclass
class DataTransformConfig:
    preprocessor_obj_path = os.path.join('artifacts','preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformConfig()
    def get_data_transformer_obj(self):
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline = Pipeline(steps = [
                ('Imputer',SimpleImputer(strategy='median')),
                ('Standardscalar',StandardScaler())
            ])
            cat_pipeline = Pipeline(steps = [
                ('Imputer',SimpleImputer(strategy = 'most_frequent')),
                ('OneHotEncoder',OneHotEncoder()),
                ('Standardscalar',StandardScaler(with_mean=False))
            ])
            preprocessor = ColumnTransformer(
                [('numpipeline',num_pipeline,numerical_columns),
                ('catpipeline',cat_pipeline,categorical_columns)
            ])
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            preprocessor_obj = self.get_data_transformer_obj()
            target_column = 'math_score'
            input_train_data = train_df.drop(columns = target_column,axis = 1)
            target_train_data = train_df[target_column]
            input_test_data = test_df.drop(columns = target_column,axis = 1)
            target_test_data = test_df[target_column]
            input_train_data_alt = preprocessor_obj.fit_transform(input_train_data)
            input_test_data_alt = preprocessor_obj.transform(input_test_data)
            train_arr = np.c_[input_train_data_alt,np.array(target_train_data)]
            test_arr = np.c_[input_test_data_alt,np.array(target_test_data)]
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_path,
                obj = preprocessor_obj
            )
            return(train_arr,test_arr,self.data_transformation_config.preprocessor_obj_path)
        except Exception as e:
            raise CustomException(e,sys)
         