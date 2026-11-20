import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info('Splitting data into train and test')
            X_train,X_test,y_train,y_test=(
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )
            
            models = {
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest": RandomForestRegressor(),
                    "Gradient Boosting": GradientBoostingRegressor(),
                    "Linear Regression": LinearRegression(),
                    "K-Neighbors Classifier": KNeighborsRegressor(),
                    "XGBClassifier": XGBRegressor(),
                    "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                    "AdaBoost Classifier": AdaBoostRegressor(),}
            
            params = {
                    "Decision Tree": {
                        "criterion": ["squared_error", "absolute_error", "poisson"],
                    },
                    "Random Forest": {
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                    },
                    "Gradient Boosting": {
                        "learning_rate": [0.1, 0.01, 0.05, 0.001],
                        "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                    },
                    "Linear Regression": {},
                    "K-Neighbors Classifier": {
                        "n_neighbors": [5, 7, 9, 11],
                    },
                    "XGBClassifier": {
                        "learning_rate": [0.1, 0.01, 0.05, 0.001],
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                    },
                    "CatBoosting Classifier": {
                        "depth": [6, 8, 10],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "iterations": [30, 50, 100],
                    },
                    "AdaBoost Classifier": {
                        "learning_rate": [0.1, 0.01, 0.5, 0.001],
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                    },}
            
            model_report=evaluate_models(models=models,X_train=X_train,X_test=X_test,y_train=y_train,y_test=y_test,params=params)
            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException('No best model found!')
            logging.info('The best model on both trainng and test data!')
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            return best_model_name,r2_square
            

        except Exception as e:
            raise CustomException(e,sys)