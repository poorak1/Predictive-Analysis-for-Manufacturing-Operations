# Predictive-Analysis-for-Manufacturing-Operations

This project is a small machine learning application that predicts the repair cost of defects based on various features such as defect type, location, severity, and inspection method. The project uses a Gradient Boosting Regressor model for prediction and provides a FastAPI-based web service for uploading data, training the model, and making predictions.

## Features

- **Data Upload**: Upload CSV files containing defect data.
- **Model Training**: Train a Gradient Boosting Regressor model using the uploaded data.
- **Cost Prediction**: Predict the repair cost for a given defect based on its features.

## Installation

1. **Dependencies**:
   ```bash
   pip install fastapi uvicorn pandas scikit-learn numpy python-multipart
   ```
## Usage

1. **Run the FASTAPI server**:
```bash
   uvicorn endpoint:app --reload
   ```
2. **Upload a CSV file**:
- Use the ```bash /upload ``` endpoint to upload a CSV file containing defect data
- Example using ```curl```:
