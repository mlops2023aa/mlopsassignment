import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
import os
import joblib

# Define the dataset path
DATASET_PATH = os.path.join("data", "car_evaluation.csv")
ENCODER_PATH = os.path.join("models", "encoder.pkl")


def check_dataset_exists(dataset_path):
    """Ensure the dataset exists."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"The dataset {dataset_path} does not exist. \
                Please ensure it's available."
        )


def load_and_preprocess_data(dataset_path):
    """Load and preprocess the dataset."""
    # Load dataset
    data = pd.read_csv(dataset_path)

    # Define column names and target column
    col_names = ['buying', 'maint', 'doors', 'persons',
                 'lug_boot', 'safety', 'class']
    data.columns = col_names
    target_column = 'class'

    # Split into features and target
    X = data.drop(columns=[target_column])  # Features
    y = data[target_column]  # Target variable

    return X, y


def split_and_encode_data(X, y):
    """Split the data into train-test sets and encode categorical variables."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Encode categorical variables
    encoder = ce.OrdinalEncoder(
        cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    )
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)

    # Save the encoder for later use
    os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"Encoder saved at {ENCODER_PATH}")

    return X_train_encoded, X_test_encoded, y_train, y_test


def get_preprocessed_data():
    """Main function to load, preprocess, and return train-test splits."""
    # Ensure dataset exists
    check_dataset_exists(DATASET_PATH)

    # Load and preprocess the dataset
    X, y = load_and_preprocess_data(DATASET_PATH)

    # Split and encode the data
    return split_and_encode_data(X, y)
