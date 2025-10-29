import os
import sys
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model = os.path.join('artifacts','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('Splitting training and test input data')
            X_train,y_train,X_test,y_test = (train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])
            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression": {
                    'fit_intercept': [True, False],
                    'positive': [False, True]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            model_report:dict = evaluate_model(x_train = X_train,y_train = y_train,x_test = X_test,y_test = y_test,models = models,params = params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            logging.info(f"Best found model on both training and testing dataset")
            save_object(
                file_path = self.model_trainer_config.trained_model,
                obj = best_model
            )
            predicted= best_model.predict(X_test)
            r2 = r2_score(y_test,predicted)
            return r2
        except Exception as e:
            logging.error(f"Error in initiate_model_trainer: {e}", exc_info=True)
            raise CustomException(e,sys)
