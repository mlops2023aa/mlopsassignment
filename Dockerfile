FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy required files
COPY requirements.txt requirements.txt
COPY app.py app.py
COPY models/tuned_models/selected_model.pkl models/selected_model.pkl
COPY models/encoder.pkl models/encoder.pkl
COPY static/ static/
COPY templates/ templates/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 5009

# Command to run the application
CMD ["python", "app.py"]

