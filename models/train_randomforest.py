from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import joblib
from preprocessing import get_preprocessed_data
import os

# Load preprocessed data
X_train, X_test, y_train, y_test = get_preprocessed_data()

with mlflow.start_run():
    # Train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100,
                                      max_depth=10,
                                      random_state=42)
    rf_model.fit(X_train, y_train)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy",
                      accuracy_score(y_test, rf_model.predict(X_test)))

    # Save the model
    MODEL_PATH = os.path.join("models", "random_forest_model.pkl")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(rf_model, MODEL_PATH)
    print(f"Random Forest model saved at {MODEL_PATH}")

    # Log the model
    mlflow.sklearn.log_model(rf_model, "random_forest_model")
