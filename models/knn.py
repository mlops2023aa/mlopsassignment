from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import mlflow
import joblib
from preprocessing import get_preprocessed_data
import os

# Load preprocessed data
X_train, X_test, y_train, y_test = get_preprocessed_data()

with mlflow.start_run():
    # Train the K-NN model
    knn_model = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
    knn_model.fit(X_train, y_train)

    # Log parameters and metrics
    mlflow.log_param("n_neighbors", 5)
    mlflow.log_param("metric", "minkowski")
    mlflow.log_param("p", 2)
    mlflow.log_metric("accuracy",
                      accuracy_score(y_test,
                                     knn_model.predict(X_test)))

    # Save the model
    MODEL_PATH = os.path.join("models", "knn_model.pkl")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(knn_model, MODEL_PATH)
    print(f"K-NN model saved at {MODEL_PATH}")

    # Log the model
    mlflow.sklearn.log_model(knn_model, "knn_model")
