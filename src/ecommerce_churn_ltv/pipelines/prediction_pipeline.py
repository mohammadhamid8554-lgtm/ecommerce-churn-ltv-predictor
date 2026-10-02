import sys
import pandas as pd
from ecommerce_churn_ltv.exception import CustomException
from ecommerce_churn_ltv.logger import logging
from ecommerce_churn_ltv.utils import load_object


class PredictPipeline:
    """Loads serialized model and preprocessor artifacts to generate real-time churn predictions."""

    def __init__(self):
        # Artifact file paths
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def predict(self, features: pd.DataFrame):
        """Loads artifacts, transforms input features, and generates churn risk prediction and probability.

        Args:
            features (pd.DataFrame): Dataframe containing raw RFM features.

        Returns:
            tuple: (prediction_class, churn_probability)
        """
        try:
            logging.info("Loading preprocessor and model artifacts for inference...")
            
            # 1. Deserialize saved model and preprocessor objects
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)

            logging.info("Transforming input customer feature matrix...")
            
            # 2. Scale input features using saved StandardScaler parameters
            data_scaled = preprocessor.transform(features)

            # 3. Predict binary churn class (0 = Retained, 1 = Churned)
            preds = model.predict(data_scaled)

            # 4. Predict exact churn risk probability percentage
            probs = model.predict_proba(data_scaled)[:, 1] if hasattr(model, "predict_proba") else [0.0]

            logging.info("Inference complete.")
            return preds[0], probs[0]

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """Maps single customer input data from API/UI requests into a structured Pandas DataFrame."""

    def __init__(self, recency: float, frequency: float, monetary: float, avg_order_value: float):
        self.recency = recency
        self.frequency = frequency
        self.monetary = monetary
        self.avg_order_value = avg_order_value

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """Converts instance attributes into a Pandas DataFrame matching model feature schema."""
        try:
            # Match the feature name used during training.
            custom_data_input_dict = {
                "Recency": [self.recency],
                "Frequency": [self.frequency],
                "Monetary": [self.monetary],
                "AveOrderValue": [self.avg_order_value],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Test sample: Customer inactive for 120 days
        sample_customer = CustomData(
            recency=120.0,
            frequency=2.0,
            monetary=150.0,
            avg_order_value=75.0
        )
        
        input_df = sample_customer.get_data_as_data_frame()
        pipeline = PredictPipeline()
        prediction, probability = pipeline.predict(input_df)

        print("\n--- PREDICTION OUTPUT ---")
        print(f"Prediction Result : {'Churn Risk' if prediction == 1 else 'Active Customer'}")
        print(f"Churn Probability : {probability * 100:.2f}%")
        print("-------------------------\n")

    except Exception as e:
        print(f"Execution Error: {e}")