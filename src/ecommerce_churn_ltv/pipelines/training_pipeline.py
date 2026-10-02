from ecommerce_churn_ltv.components.data_ingestion import DataIngestion
from ecommerce_churn_ltv.components.data_transformation import DataTransformation
from ecommerce_churn_ltv.components.model_trainer import ModelTrainer

def run_training_pipeline() -> float:
	train_path, test_path = DataIngestion().initiate_data_ingestion()
	train_array, test_array, _ = DataTransformation().initiate_data_transformation(
		train_path, test_path
	)
	return ModelTrainer().initiate_model_trainer(train_array, test_array)

if __name__ == "__main__":
	score = run_training_pipeline()
	print(f"Training completed. Best model ROC-AUC: {score:.4f}")
