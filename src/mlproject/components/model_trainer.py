import os
import sys
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import r2_score
from dataclasses import dataclass
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object
from src.mlproject.utils import evaluate_models
from src.mlproject.components.data_transformation import DataTransformation
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=0),
            }
            param = {
                "Random Forest": {},
                "Linear Regression": {},
                "K-Neighbors Regressor": {},
                "XGBRegressor": {},
                "CatBoosting Regressor": {}
            }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models,param)

            # Get the best model
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            best_score = model_report[best_model_name]

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_score}")

            if best_score < 0.6:
                raise CustomException("No suitable model found with acceptable accuracy.", sys)

            # Fit the best model
            best_model.fit(X_train, y_train)

            # Save model locally
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            joblib.dump(best_model, self.model_trainer_config.trained_model_file_path)

            # Log model artifact to MLflow (Dagshub)
            mlflow.log_artifact(self.model_trainer_config.trained_model_file_path)

            # Evaluate and log score
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            mlflow.log_metric("r2_score", r2)

            return self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)



'''import os
import sys
from dataclasses import dataclass
from urllib.parse import urlparse
import numpy as np

import mlflow
import mlflow.sklearn
import dagshub  # ✅ Dagshub import

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print(f" Best Model: {best_model_name}")

            actual_model = best_model_name
            best_params = params.get(actual_model, {})

            # ✅ Initialize Dagshub
            dagshub.init(repo_owner='neeraj015', repo_name='neerajDS_project1', mlflow=True)

            # ✅ Setup MLflow
            mlflow.set_registry_uri("https://dagshub.com/neeraj015/neerajDS_project1.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            with mlflow.start_run():
                predicted = best_model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, predicted)

                mlflow.log_params(best_params)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
               
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")
            
                mlflow.sklearn.log_model(best_model, "model")

            if best_model_score < 0.6:
                raise CustomException("❌ No good model found with acceptable performance")

            logging.info("Saving the best model to disk")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return r2

        except Exception as e:
            raise CustomException(e, sys)
'''