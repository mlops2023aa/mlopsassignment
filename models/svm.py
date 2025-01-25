from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import mlflow
import joblib
from preprocessing import get_preprocessed_data
import os

# Load preprocessed data
X_train, X_test, y_train, y_test = get_preprocessed_data()

with mlflow.start_run():
    # Train the SVM model
    svm_model = SVC(kernel="rbf", C=1.0, gamma="scale")
    svm_model.fit(X_train, y_train)

    # Log parameters and metrics
    mlflow.log_param("kernel", "rbf")
    mlflow.log_param("C", 1.0)
    mlflow.log_param("gamma", "scale")
    mlflow.log_metric("accuracy", accuracy_score(y_test, svm_model.predict(X_test)))

    # Save the model
    MODEL_PATH = os.path.join("models", "svm_model.pkl")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(svm_model, MODEL_PATH)
    print(f"SVM model saved at {MODEL_PATH}")

    # Log the model
    mlflow.sklearn.log_model(svm_model, "svm_model")
