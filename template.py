import logging
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)


project_name = "ecommerce_churn_ltv"

list_of_files = [

    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitoring.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "app.py",
    "Dockerfile",
    "main.py",
    "setup.py",
    "requirements.txt"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Create Directory: {filedir}")

    if (not os.path.exists(filepath) or os.path.getsize(filepath)):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename}file is already exists!!")