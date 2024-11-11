import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2_square = r2_score(true, predicted)

    return mae, rmse, r2_square

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Evaluates multiple models on training and testing datasets, logs performance metrics,
    and returns a report dictionary with model names and metrics.

    Parameters:
    - models (dict): A dictionary of model names as keys and model instances as values.
    - X_train, X_test: Training and testing feature sets.
    - y_train, y_test: Training and testing target variables.

    Returns:
    - dict: A report dictionary with model performance metrics for both train and test sets.
    
    Raises:
    - CustomException: If an error occurs during model evaluation, it raises a custom exception with the error details.
    """
    try:
        method_name = f"{evaluate_models.__module__}.{evaluate_models.__name__}"
        logging.info(f"{method_name} - Starting model evaluation.")

        report = {}

        for name, model in models.items():
            logging.info(f"{method_name} - Training and evaluating model: {name}")
            
            # Fit the model
            model.fit(X_train, y_train)
            logging.info(f"{method_name} - Model {name} training completed.")

            # Predict on train and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate model performance on train and test sets
            model_train_mae, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
            model_test_mae, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

            logging.info(f"{method_name} - Model {name} performance on training set: "
                         f"RMSE={model_train_rmse:.4f}, MAE={model_train_mae:.4f}, R2={model_train_r2:.4f}")
            logging.info(f"{method_name} - Model {name} performance on test set: "
                         f"RMSE={model_test_rmse:.4f}, MAE={model_test_mae:.4f}, R2={model_test_r2:.4f}")

            # Store metrics in report dictionary
            report[name] = {
                'Train': {
                    'MAE': model_train_mae,
                    'RMSE': model_train_rmse,
                    'R2': model_train_r2,
                },
                'Test': {
                    'MAE': model_test_mae,
                    'RMSE': model_test_rmse,
                    'R2': model_test_r2,
                }
            }

            # Print model performance for logging in console
            print(f"Model: {name}")
            print('Model performance for Training set')
            print(f"- Root Mean Squared Error: {model_train_rmse:.4f}")
            print(f"- Mean Absolute Error: {model_train_mae:.4f}")
            print(f"- R2 Score: {model_train_r2:.4f}")
            print('----------------------------------')
            print('Model performance for Test set')
            print(f"- Root Mean Squared Error: {model_test_rmse:.4f}")
            print(f"- Mean Absolute Error: {model_test_mae:.4f}")
            print(f"- R2 Score: {model_test_r2:.4f}")
            print('='*35)
            print('\n')
        
        logging.info(f"{method_name} - Model evaluation completed.")
        return report

    except Exception as e:
        logging.error(f"Error in {method_name}: {e}")
        raise CustomException(e, sys)

