from setuptools import setup, find_packages
from pathlib import Path

# Constant used in requirements.txt to trigger local package installation in editable mode(-e .)

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> list[str]:
    """
    Reads the requirements.txt file, cleans whitespace/newlines,
    and returns a clean list of dependencies for setuptools.

    """

    requirements = []
    path = Path(file_path)

    if path.exists():
        with open(path, "r", encoding="utf-8") as file_obj:
            # reads all the files from requirements.txt
            requirements = file_obj.readlines()

            # Clean up trailing line break (\n) and extra whitespace
            requirements = [req.replace("\n", "") for req in requirements]

            # Remove -e . from the list so setuptools doesnn't get confused 
            # (-e . in only meant for pip, not setuptools install_requires argument)
            if HYPHEN_E_DOT in requirements:
                requirements.remove(HYPHEN_E_DOT)
    return requirements

# Metadata and build configuration for the package

setup(

    name= "ecommerce_churn_ltv",
    version="0.0.1",
    author= "Mohammed Hamid",
    author_email= "mohammadhamid8554@gmail.com",
    packages= find_packages(),
    install_requires= get_requirements("requirements.txt")
)