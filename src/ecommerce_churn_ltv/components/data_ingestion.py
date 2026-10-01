# Import Required Libraries

import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from ecommerce_churn_ltv.exception import CustomException
from ecommerce_churn_ltv.logger import logging


@dataclass
class DataIngestionConfig:
    """Config class to store artifacts file output for the ingestion step"""

    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")


class DataIngestion:
    """
    Handles loading raw data,  saving raw artifacts, and generating train/test splits.
    generating RFM + Churn features.

    """

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def _process_transactional_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw transaction logs into customero-level RFM and churn data."""
        # 1. Clean data: drop missing Customer ID & negative quantities/prices
        df = df.dropna(subset=["Customer ID"]).copy()
        df["Customer ID"] = df["Customer ID"].astype(int).astype(str)
        df = df[(df["Quantity"] > 0) & (df["Price"] > 0)].copy()

        # 2. Convert Invoice to datetime & Calculate Total Spend
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["TotalSpend"] = df["Quantity"] * df["Price"]

        # 3. Snapshot date for recency calculation (day after last transaction)
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

        # 4. Aggregate to Customer Level (RFM)
        customer_df = df.groupby("Customer ID").agg(
            Recency = ("InvoiceDate", lambda x:(snapshot_date - x.max()).days),
            Frequency = ("Invoice", "nunique"),
            Monetary = ("TotalSpend", "sum"),
            AveOrderValue = ("TotalSpend", "mean")
        )

        # 5. Define Churn: Customer has not purchased in the last 90 days
        customer_df["churn"] = (customer_df["Recency"] > 90).astype(int)

        # 6. Define 6-Month LTV proxy(TotalSpend)
        customer_df["ltv"] = customer_df["Monetary"]

        return customer_df
 
    def initiate_data_ingestion(self) -> tuple[str, str]:
        """
        Reads raw customer data, creates artifacts directory, splits into train/test
        and returns paths to train and test CSV files.

        """
        logging.info("Starting Data Ingestion component execution....")
        try:
            # Path to the raw input CSV(created earlier in data/ ro processed from Kaggle)
            source_data_path = os.path.join("data", "ecom_data.csv")

            if not os.path.exists(source_data_path):
                raise FileNotFoundError(
                    f"Source data file [{source_data_path}] not found."
                    f"Ensure dataset exists in data/directory."
                )

            # Read dataset into pandas DataFrame
            df_raw = pd.read_csv(source_data_path, encoding="ISO-8859-1")
            logging.info(f"Loaded source dataset successfully. Shape: {df_raw.shape}")

            # create artifacts directory if missing 
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            # save raw copy into artifacts directory for audit/reproducibility
            df_raw.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved to: {self.ingestion_config.raw_data_path}")

            # Process transaction into Customer-Level dataset with "churn" label
            logging.info("Processing transactions into customer-level RFM & Churn features...")
            df_customer = self._process_transactional_data(df_raw)
            logging.info(f"Customer dataset generated. Shape: {df_customer.shape}")

            # Train-Test Split (80% train, 20% test, stratified on churn target)
            logging.info("Splitting dataset into train and test sets....")
            train_set, test_set = train_test_split(
                df_customer,
                random_state=42,
                test_size=0.2,
                stratify=df_customer["churn"],
            )


            # Save train and test sets into artifacts folder 
            train_set.to_csv(
                self.ingestion_config.train_data_path, index = False, header = True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path, index = False, header = True
            )

            logging.info("Data Ingestion completed successfully!")
            logging.info(f"Train path: {self.ingestion_config.train_data_path}")
            logging.info(f"Test path: {self.ingestion_config.test_data_path}")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
if __name__ == "__main__":
        obj = DataIngestion()
        train_path, test_path = obj.initiate_data_ingestion()