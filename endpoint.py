from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
from io import StringIO
import pickle
from call import train
import numpy as np

app = FastAPI()

files = []

@app.get("/")
def root():
    return {"Hello":"World"}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a valid CSV file.")

    try:
        content = await file.read()
        file_name = file.filename
        files.append(file_name)
        df = pd.read_csv(StringIO(content.decode('utf-8')))
        df.to_csv(file_name)
        return {"message": "File processed successfully", "File Name": file_name, "columns": df.columns.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
@app.post("/train")
async def train_model(csv_name: str):
    if(csv_name in files):
        error = train(csv_name)
    else:
        return {"CSV not found, Kindly upload the CSV using upload endpoint"}
    return {"message": "Training Complete", "Mean Squared Error": error}

@app.get("/predict")
async def get_prediction(id: int, type: str, location: str, severity: str, inspection_method: str):
    with open('GB_Boost_Classifier.pkl', 'rb') as f:
        model = pickle.load(f)

    with open("inspection_method_encoder.pkl", "rb") as f:
        type_encoder = pickle.load(f)

    with open("defect_location_encoder.pkl", "rb") as f:
        location_encoder = pickle.load(f)

    with open("inspection_method_encoder.pkl", "rb") as f:
        inspection_method_encoder = pickle.load(f)

    with open("severity_encoder.pkl", "rb") as f:
        severity_encoder = pickle.load(f)

    type_encoded = type_encoder.transform(np.array([type]).reshape(-1, 1)).toarray() \
        if type in type_encoder.categories_[0] else np.zeros((1, len(type_encoder.categories_[0])))
    location_encoded = location_encoder.transform(np.array([location]).reshape(-1, 1)).toarray() \
        if location in location_encoder.categories_[0] else np.zeros((1, len(location_encoder.categories_[0])))
    inspection_method_encoded = inspection_method_encoder.transform(np.array([inspection_method]).reshape(-1, 1)).toarray() \
        if inspection_method in inspection_method_encoder.categories_[0] else np.zeros((1, len(inspection_method_encoder.categories_[0])))

    severity_encoded = severity_encoder.transform([severity]) \
        if severity in severity_encoder.classes_ else [0]

    features = np.concatenate([
        [id],  
        type_encoded.flatten(),  
        location_encoded.flatten(),
        inspection_method_encoded.flatten(),
        severity_encoded
    ])
    
    predicted_cost = model.predict([features])[0]
    return {"Predicted Repair Cost": predicted_cost}