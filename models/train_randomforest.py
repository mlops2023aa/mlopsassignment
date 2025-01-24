import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import category_encoders as ce
import joblib
import os
import mlflow
import mlflow.sklearn

# Path to the dataset
DATASET_PATH = os.path.join("data", "car_evaluation.csv")

# Check if the dataset exists
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"""The dataset {DATASET_PATH} does not exist.
          Please ensure it's available.""")

# Step 1: Load the dataset
data = pd.read_csv(DATASET_PATH)

# Step 2: Basic preprocessing (modify as per your dataset)
# view dimensions of dataset
data.shape

# preview the dataset
data.head()
col_names = ['buying', 'maint', 'doors', 'persons',
             'lug_boot', 'safety', 'class']
data.columns = col_names
col_names

# let's again preview the dataset
data.head()

data.info()

col_names = ['buying', 'maint', 'doors', 'persons',
             'lug_boot', 'safety', 'class']
for col in col_names:
    print(data[col].value_counts())

data['class'].value_counts()

# check missing values in variables
data.isnull().sum()

target_column = 'class'
if target_column not in data.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset.")

X = data.drop(['class'], axis=1)  # Features
y = data[target_column]  # Target variable

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

X_train.shape, X_test.shape
X_train.dtypes

X_train.head()

# encode categorical variables with ordinal encoding
encoder = ce.OrdinalEncoder(cols=['buying', 'maint',
                                  'doors', 'persons', 'lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

X_train.head()
# Start an MLflow run
with mlflow.start_run():

    # Step 3: Train a RandomForest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # Log model parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("random_state", 42)

    # Step 4: Evaluate the model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Step 5: Save the model
    MODEL_PATH = os.path.join("models", "random_forest_model.pkl")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(rf_model, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")

    # Log the model to MLflow
    mlflow.sklearn.log_model(rf_model, "random_forest_model")

    # Optionally log the dataset used
    mlflow.log_artifact(DATASET_PATH, artifact_path="data")

# End of MLflow run
