import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from src.logger import logging
from sklearn.model_selection import GridSearchCV

def load_object(file_path):
        try:
            with open(file_path,'rb') as file_obj:
                return dill.load(file_obj)
        except Exception as e:
            raise CustomException(e,sys)
            

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
def evaluate_model(x_train,y_train,x_test,y_test,models,params):
    try:
        report = {}
        for i in range(len(models)):
            logging.info(f"Training model: {list(models.values())[i]}")
            model = list(models.values())[i]
            param = list(params.values())[i]
            gs = GridSearchCV(param_grid = param,cv = 3,n_jobs = 3,estimator=model)
            gs.fit(x_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            y_test_pred = model.predict(x_test)
            y_train_pred = model.predict(x_train)
            test_score = r2_score(y_test,y_test_pred)
            report[list(models.keys())[i]] = test_score
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
    