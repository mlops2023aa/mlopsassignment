import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..')))
from models.preprocessing import get_preprocessed_data  # noqa: E402
import pandas as pd


# Test preprocessing - check if data is loaded and preprocessed properly
def test_get_preprocessed_data():
    data = get_preprocessed_data()
    # Check that the dataset is a pandas DataFrame
    assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame"
    # Check that the target column is present
    assert 'class' in data.columns, "Target column 'class' should be present"

    # Check for missing values
    assert data.isnull().sum().sum() == 0, "Data contains missing values"
    # Check shape of the dataset (adjust based on expected shape)
    assert data.shape[0] > 0, "Dataset should not be empty"
    assert data.shape[1] == 7, "Expected 7 columns in the dataset"
