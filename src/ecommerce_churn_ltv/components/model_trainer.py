import os
import sys
from dataclasses import dataclass

# Import candidate ML algorithms (Tree ensemble + Linear baselin)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Import custom logging, exception, and saving utilities
from ecommerce_churn_ltv.logger import logging
from ecommerce_churn_ltv.exception import CustomException
from ecommerce_churn_ltv.utils import save_object, evaluat_models

# 1. Configuration: Define where the winning model artifacts(.pkl) will be saved
@dataclass
class ModelTrainingConfig:
    """Stores output path for the best model artifacts."""
    trained_model_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    """Trains multiple models, evaluates metrics, and exports the top performance."""

    def __init__(self):
        # Initialize configuration with default artifacts target path
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_array, test_array) -> float:
        """
        
        Splits tranformed arrays into X/y sets, runs model evaluation, 
        and exports the best model artifacts

        """

        try:
            logging.info("Splitting transformed train and test array into features and targets...")

            # 2. Extract Features (X)  and Target Labels (y)  from preprocessor Numpy arrays
            # Array slice [:, :-1] grabs all columns except the last one (features)
            # Array slice [:, -1] grabs only the last column (churn label: 0 or 1)

            X_train, y_train, X_test, y_test = (train_array[:, :-1],
                                                train_array[:, -1],
                                                test_array[:, :-1],
                                                test_array[:, -1]
                                                )
            # 3. Define the dictionary of candidata classifiers to evaluate

            models = {

                "Random Forest" : RandomForestClassifier(random_state=42),
                "Gradient Boosting": XGBClassifier(random_state = 42),
                "Logistic Regression" : LogisticRegression(random_state=42),
                "XGBoost":  XGBClassifier(use_label_encoder = False, eval_metric = "logloss", random_state = 42),
                "LightGBM" : LGBMClassifier(random_state=42, verbose = -1)
            }

            logging.info("Evaluating candidate classification models....")

            # 4. Pass candidate to evaluate_models() helper (from utils.py)
            # First each model on X_trian/y_train and calcualtes ROC_AUC score on X_test/y_test

            model_report:dict = evaluat_models(
                X_train= X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            # 5. Extract the top-performing model score and its model name
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)] 

            best_model = models[best_model_name]

            # 6. Quality Gae: Throw an exception if no model meets the threshold score of o.60
            if best_model_score < 0.60:
                raise CustomException(Exception("NO suitable model found with ROC_AUC >= 0.60"), sys)

            logging.info("Best Model Found: [{best_model_name}] with Test ROC-AUC Score: {best_model_score:.4f}")

            # 7. Serializse and export the winning model object as 'model.pkl' inside artifacts/

            save_object(file_path=self.model_trainer_config.trained_model_path, obj=best_model)

            return best_model_score
        
        except Exception as e:
            raise CustomException(e, sys)