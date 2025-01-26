# mlopsassignment
Repository for mlops assignment
## Project Setup

### Prerequisites
- Python 3.8+
- Git installed
-The integration of DVC and google drive is broken. You will need to copy the data file from
https://drive.google.com/file/d/1JOQfclYuxeguyPbEyXhQLFv4v7wBLTdh/view?usp=sharing to /data folder as a pre-requsite for the project

### Steps to Set Up the Project
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>

### Steps to Install dependencies
pip install -r requirements.txt

### Steps to Initialize DVC
dvc init
dvc pull
-Make sure you have access to the configured DVC remote

### Run MLflow UI
mlflow ui
