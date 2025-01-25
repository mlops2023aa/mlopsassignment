from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from preprocessing import get_preprocessed_data
import pandas as pd
import os
import joblib
import mlflow

# Paths
DATASET_PATH = os.path.join("data", "car_evaluation.csv")
MODEL_OUTPUT_DIR = os.path.join("models", "tuned_models")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Preprocess the data
X_train, X_test, y_train, y_test = get_preprocessed_data()


# Function to perform GridSearchCV for any model
def perform_hyperparameter_tuning(model, param_grid, model_name):
    with mlflow.start_run(run_name=f"{model_name}_Hyperparameter_Tuning"):
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=3,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_

        # Evaluate the best model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"Best Parameters for {model_name}: {best_params}")
        print(f"Accuracy for {model_name}: {accuracy:.2f}")
        print(f"Confusion Matrix for {model_name}:\n{cm}")

        # Log results in MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", accuracy)

        # Save the best model
        model_path = os.path.join(MODEL_OUTPUT_DIR,
                                  f"{model_name}_best_model.pkl")
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path, artifact_path=f"{model_name}_models")
        print(f"Best {model_name} model saved to {model_path}\n")

        return best_model, best_params


# Define parameter grids for each model
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

knn_param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Perform hyperparameter tuning for each model
print("Tuning Random Forest...")
rf_best_model, rf_best_params = perform_hyperparameter_tuning(
    RandomForestClassifier(random_state=42), rf_param_grid, "Random_Forest"
)

print("Tuning K-Nearest Neighbors...")
knn_best_model, knn_best_params = perform_hyperparameter_tuning(
    KNeighborsClassifier(), knn_param_grid, "KNN"
)

print("Tuning Support Vector Machine...")
svm_best_model, svm_best_params = perform_hyperparameter_tuning(
    SVC(random_state=42), svm_param_grid, "SVM"
)

# Load the tuned models
rf_model_path = os.path.join(MODEL_OUTPUT_DIR, "Random_Forest_best_model.pkl")
knn_model_path = os.path.join(MODEL_OUTPUT_DIR, "KNN_best_model.pkl")
svm_model_path = os.path.join(MODEL_OUTPUT_DIR, "SVM_best_model.pkl")

rf_model = joblib.load(rf_model_path)
knn_model = joblib.load(knn_model_path)
svm_model = joblib.load(svm_model_path)

# Dictionary to store metrics for each model
model_metrics = {}

# Evaluate each model on the test data
for model_name, model in [("Random Forest", rf_model),
                          ("KNN", knn_model),
                          ("SVM", svm_model)]:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    # Store metrics
    model_metrics[model_name] = {
        "Accuracy": accuracy,
        "Classification Report": report,
        "Confusion Matrix": cm
    }
    # Display results
    print(f"--- {model_name} ---")
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(pd.DataFrame(report).transpose())
    print("\n")

# Determine the best model based on accuracy
best_model_name = max(model_metrics,
                      key=lambda x: model_metrics[x]["Accuracy"])
best_model = {
    "Random Forest": rf_model,
    "KNN": knn_model,
    "SVM": svm_model
}[best_model_name]

print(f"""The best model is: {best_model_name} with Accuracy:
       {model_metrics[best_model_name]['Accuracy']:.2f}""")
