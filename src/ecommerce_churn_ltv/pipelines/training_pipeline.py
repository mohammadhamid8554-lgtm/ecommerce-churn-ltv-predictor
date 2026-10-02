import sys
from ecommerce_churn_ltv.exception import CustomException
from ecommerce_churn_ltv.logger import logging
from ecommerce_churn_ltv.components.data_ingestion import DataIngestion
from ecommerce_churn_ltv.components.data_transformation import DataTransformation
from ecommerce_churn_ltv.components.model_trainer import ModelTrainer


class TrainPipeline:
    """Automates the full training lifecycle: Data Ingestion -> Data Transformation -> Model Training."""

    def __init__(self):
        pass

    def run_pipeline(self) -> float:
        """Executes each component sequentially and returns the final model performance metric."""
        try:
            logging.info("==========================================")
            logging.info("Starting End-to-End Training Pipeline Execution...")
            logging.info("==========================================")

            # 1. Component 1: Data Ingestion
            logging.info("Pipeline Execution: Running Data Ingestion...")
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            # 2. Component 2: Data Transformation
            logging.info("Pipeline Execution: Running Data Transformation...")
            data_transformation = DataTransformation()
            train_arr, test_arr, preprocessor_path = (
                data_transformation.initiate_data_transformation(
                    train_path=train_path, test_path=test_path
                )
            )

            # 3. Component 3: Model Training
            logging.info("Pipeline Execution: Running Model Trainer...")
            model_trainer = ModelTrainer()
            best_model_score = model_trainer.initiate_model_trainer(
                train_array=train_arr, test_array=test_arr
            )

            logging.info("==========================================")
            logging.info(
                f"Training Pipeline Completed Successfully! Final ROC-AUC: {best_model_score:.4f}"
            )
            logging.info("==========================================")

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    roc_auc = pipeline.run_pipeline()
    print(f"Training Pipeline Run Finished. Best Model Score: {roc_auc:.4f}")