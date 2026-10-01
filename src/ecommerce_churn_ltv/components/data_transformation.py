import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from ecommerce_churn_ltv.exception import CustomException
from ecommerce_churn_ltv.logger import logging
from ecommerce_churn_ltv.utils import save_object

@dataclass
class DataTransformationConfig:
    """Stores output artifacts path fro preprocessor pickle file."""
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    """Handles feature scaling, preprocessing pipeline construction, and SMOTE oversampling."""

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self) -> ColumnTransformer:
        """Creates and returns the Scikit-learn ColumnTransformation pipeline for numeric scaling."""
        try:
            numerical_columns = ["Recency", "Frequency", "Monetary", "AveOrderValue"]

            # Define pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            logging.info(f"Numerical columns to scale: {numerical_columns}")

            # Assemble ColumnTransformation
            preprocessor = ColumnTransformer(transformers = [
                ("num_pipeline", num_pipeline, numerical_columns)
            ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
            self, train_path: str, test_path: str
    ) -> tuple[np.ndarray, np.ndarray, str]:

        """
            Transforms raw train/test features. balances classes with SMOTE, and saves preprocessor object.
            Returns:
                tuple: (train_array, test_array, preprocessor_pickle_file_path)
        """
        try:
            """Read train and test dataset from artifacts/"""
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data successfully for transformation.")

            # Retrive preprocessor object instance
            preprocessor_obj = self.get_data_transformation_object()

            target_column_name = "churn"
            ignore_columns = ["ltv", target_column_name]

            # Separate feature matrix (X) and target variable(y)
            input_feature_train_df = train_df.drop(columns=ignore_columns)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=ignore_columns)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessor object on training and testing dataframes.")

            # Fit-Transform training data; transform test data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Apply SMOTE to handle class imbalance on training set only
            logging.info("Applying SMOTE oversampling to training dataset...")
            smote = SMOTE(random_state=42)
            input_feature_train_arr, target_feature_train_df = smote.fit_resample( 
                input_feature_train_arr, target_feature_train_df
            )

            # Combine transformed features and target into numpy array

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            # Save serializd preprocessor object
            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path, obj = preprocessor_obj)


            logging.info(f"Saved preprocessor object to: {self.data_transformation_config.preprocessor_obj_file_path}")

            return(train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from ecommerce_churn_ltv.components.data_ingestion import DataIngestion

    # Run Data Ingestion first to ensure paths exist
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Execute Data Transformation
    transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformation.initiate_data_transformation(
        train_path, test_path
    )
    print("Transformation completed successfully!")