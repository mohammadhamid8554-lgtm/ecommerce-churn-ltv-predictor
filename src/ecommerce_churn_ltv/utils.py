import os
import sys
import pickle

from ecommerce_churn_ltv.logger import logging
from ecommerce_churn_ltv.exception import CustomException
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def save_obj(file_path: str, obj: object) -> None:
    """
    Saves a Python object(e.g., preprocessor, ML model) as a serialized pickle (.pkl) file.

    Args: 
        file_path(str): Destination path where the object should be saved.
        obj (object): The Python object to serialize.
    
    """
    try:
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Successfully saved object to path: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str) ->object:
    """
    Loads a serialized pickle (.pkl) file from disk into memory.

    Agrs:
        file_path(str): Path to the saed pickle file.
    
    Returns:
        Object: Deserialized Python object.

    """

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        with open(file_path, "rb")  as file_obj:
            obj = pickle.load(file_obj)

        logging.info(f"Successfully loaded object from path: {file_path}")
        return obj

    except Exception as e:
        raise CustomException(e, sys)

def evaluat_models(X_train, y_train, X_test, y_test, models: dict) -> dict:

    """
    Trains and evaluaes multiple canditae classification models.

    Args: 
        X_train, y_train =  Training features and labels
        X_test, y_test  = Testing features and labels
        model (dict): Dictionary where keys are model names and values are model instances.

    Returns:
        dict: Report mapping model names to their test ROC-AUC scores.
    """

    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            # Train model
            model.fit(X_train, y_train)

            # Generate predictions
            y_test_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_test_proba = model.predict_proba(X_test)[:, 1]
                test_roc_auc = roc_auc_score(y_test, y_test_proba)
            elif hasattr(model, "decision_function"):
                test_scores = model.decision_function(X_test)
                test_roc_auc = roc_auc_score(y_test, test_scores)
            else:
                raise ValueError(
                    f"Model [{model_name}] must provide predict_proba or decision_function "
                    "to calculate ROC-AUC."
                )

            test_f1 = f1_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)

            logging.info(
                "Model [%s] Performance - ROC-AUC: %.4f, F1: %.4f, Precision: %.4f, Recall: %.4f",
                model_name,
                test_roc_auc,
                test_f1,
                test_precision,
                test_recall,
            )

            report[model_name] = test_roc_auc

        return report

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    test_dict = {"project": "ecommerce_churn_ltv", "version": "1.0"}
    test_path = "artifacts/test_utils.pkl"

    # Test saving
    save_obj(test_path, test_dict)

    # Test loading
    loaded_data = load_object(test_path)
    print("Loaded test object:", loaded_data)

    # Clean up test artifact
    if os.path.exists(test_path):
        os.remove(test_path)