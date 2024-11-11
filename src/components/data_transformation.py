import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation that defines the path to save 
    the preprocessor object.
    """
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    """
    Class for handling data transformations, including scaling for numerical columns 
    and encoding for categorical columns.
    """
    
    def __init__(self):
        """
        Initializes the DataTransformation class with a configuration object.
        """
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self, train_df, target_column_name) -> ColumnTransformer:
        """
        Creates a preprocessor object that applies transformations to numerical 
        and categorical columns. Numerical columns are imputed and scaled, while 
        categorical columns are imputed and one-hot encoded.
        
        Returns:
        - ColumnTransformer: A preprocessor object for transforming data.
        
        Raises:
        - CustomException: If any error occurs during the creation of the transformer.
        """
        try:
            # Define columns to transform
            numerical_columns = train_df.select_dtypes(exclude=['object', 'category']).columns.tolist()
            categorical_columns = train_df.select_dtypes(include=['object', 'category']).columns.tolist()

            # Remove target column from numerical column if present
            if target_column_name in numerical_columns:
                numerical_columns.remove(target_column_name)

            # Remove target column from categorical column if present
            if target_column_name in categorical_columns:
                categorical_columns.remove(target_column_name)
            
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            # Pipeline for numerical columns: median imputation and standard scaling
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Pipeline for categorical columns: frequent category imputation and one-hot encoding
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoder', OneHotEncoder())
                ]
            )

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")
        
            # Column transformer combining both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            logging.error("Error in get_data_transformer_obj: %s", e)
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Reads training and testing data from CSV files, applies preprocessing transformations
        on numerical and categorical columns, and saves the transformation object for future use.
        
        Parameters:
        - train_path (str): Path to the CSV file containing the training dataset.
        - test_path (str): Path to the CSV file containing the testing dataset.
        
        Returns:
        - tuple: A tuple containing:
            - train_arr (np.ndarray): Transformed training data with combined features and target.
            - test_arr (np.ndarray): Transformed testing data with combined features and target.
            - preprocessor_obj_path (str): Path where the preprocessor object is saved.
            
        Raises:
        - CustomException: If an error occurs during data transformation, it raises a custom exception with the error details.
        
        Notes:
        - The method currently uses "math_score" as the hardcoded target column. 
        To make it more flexible, consider passing the target column name as a parameter.
        """
        try:
            method_name = f"{self.__class__.__name__}.{sys._getframe().f_code.co_name}"
            
            # Load training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"{method_name} - Successfully read train and test data.")

            # Obtain the target column and preprocessing object
            target_column_name = "math_score"  # TODO: Remove hardcoding
            logging.info(f"{method_name} - Obtaining preprocessing object.")
            preprocessing_obj = self.get_data_transformer_obj(train_df, target_column_name)

            # Separate input features and target for training and testing data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"{method_name} - Applying preprocessing on training and testing data.")

            # Apply transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)  # Changed to transform for test data only

            # Combine transformed features with the target variable
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessing object
            logging.info(f"{method_name} - Saving preprocessing object.")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info(f"{method_name} - Data transformation complete.")
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.error(f"Error in {method_name}: {e}")
            raise CustomException(e, sys)
