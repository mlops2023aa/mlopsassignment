import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
import os

# Path to the dataset
DATASET_PATH = os.path.join("data", "car_evaluation.csv")

# Check if the dataset exists
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"""The dataset {DATASET_PATH} does not exist.
          Please ensure it's available."""
    )

# Step 1: Load the dataset
data = pd.read_csv(DATASET_PATH)

# Basic preprocessing
col_names = ['buying', 'maint', 'doors',
             'persons', 'lug_boot', 'safety', 'class']
data.columns = col_names

# Split into features and target
X = data.drop(['class'], axis=1)  # Features
y = data['class']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Encode categorical variables
encoder = ce.OrdinalEncoder(cols=['buying', 'maint',
                                  'doors', 'persons', 'lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)


# Export preprocessed data
def get_preprocessed_data():
    return X_train, X_test, y_train, y_test
