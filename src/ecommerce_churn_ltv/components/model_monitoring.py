import os
import sys
from typing import Any, cast

import pandas as pd
from scipy.stats import ks_2samp

from ecommerce_churn_ltv.exception import CustomException
from ecommerce_churn_ltv.logger import logging


class ModelMonitoring:
    """Monitors incoming evaluation/production data against training baselines

    to detect feature distribution drift using statistical testing.
    """

    def __init__(self):
        pass

    def detect_dataset_drift(
        self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05
    ) -> bool:
        """Compares baseline training data with incoming data using two-sample KS test.

        Args:
            base_df (pd.DataFrame): Original training baseline dataset.
            current_df (pd.DataFrame): Incoming dataset for validation/prediction.
            threshold (float): Significance level (p-value threshold) for drift detection.

        Returns:
            bool: True if data drift is detected in any feature, False otherwise.
        """
        try:
            drift_detected = False
            numerical_features = ["Recency", "Frequency", "Monetary", "AvgOrderValue"]

            logging.info("Starting Kolmogorov-Smirnov statistical data drift check...")

            for feature in numerical_features:
                if feature in base_df.columns and feature in current_df.columns:
                    # Perform two-sample Kolmogorov-Smirnov test
                    # Null Hypothesis (H0): Both samples come from the same distribution
                    ks_result = ks_2samp(base_df[feature], current_df[feature])
                    ks_stat = ks_result.statistic
                    p_value = ks_result.pvalue

                    # If p-value < threshold, reject H0 -> Distributions are significantly different
                    if p_value < threshold:
                        logging.warning(
                            f"Drift detected in feature [{feature}]! "
                            f"KS-Statistic: {ks_stat:.4f}, p-value: {p_value:.4f}"
                        )
                        drift_detected = True
                    else:
                        logging.info(
                            f"No drift detected in feature [{feature}]. "
                            f"p-value: {p_value:.4f}"
                        )

            if drift_detected:
                logging.warning("Alert: Dataset drift detected across incoming batch.")
            else:
                logging.info("Dataset drift check passed. Distributions match baseline.")

            return drift_detected

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Load baseline train set and test set to verify drift checking logic
        train_df = pd.read_csv(os.path.join("artifacts", "train.csv"))
        test_df = pd.read_csv(os.path.join("artifacts", "test.csv"))

        monitor = ModelMonitoring()
        has_drift = monitor.detect_dataset_drift(base_df=train_df, current_df=test_df)
        print(f"Drift Check Execution Complete. Drift Detected: {has_drift}")

    except Exception as e:
        raise CustomException(e, sys)