from fastapi import FastAPI, HTTPException
from joblib import load
import numpy as np

app = FastAPI(
    title="Alzheimer's Disease Detection API",
    version="1.0",
    description=(
        "This API provides predictions for Alzheimer's disease based on various health metrics. "
        "To use this API, send a POST request to the /prediction endpoint with the following parameters:\n\n"
        "- **ADL**: Level of difficulty in Activities of Daily Living (0: No difficulty, 4: Complete difficulty)\n"
        "- **FunctionalAssessment**: Functional Assessment score (0: No impairment, 4: Very severe impairment)\n"
        "- **MMSE**: Mini-Mental State Examination score (0: Severely impaired, 3: Normal)\n"
        "- **BehavioralProblems**: Presence of behavioral problems (0: No, 1: Yes)\n"
        "- **MemoryComplaints**: Presence of memory complaints (0: No, 1: Yes)\n"
        "- **SleepQuality**: Sleep Quality (4: Barely got any sleep, 10: 8 hours per day)\n"
        "- **CholesterolHDL**: Cholesterol HDL level (20: Very low, 70: Extremely high)\n\n"
        "The API will return a prediction indicating the likelihood of Alzheimer's disease based on the provided inputs.\n\n\n"
        
        "prediction: 1 means You are Alzheimer.\n\n"
        "prediction: 0 means You are not Alzheimer."
    )
)

# Define the list of features (ensure the order matches the model's expected input)
features = ['ADL', 'FunctionalAssessment', 'MMSE', 'BehavioralProblems', 'MemoryComplaints', 'SleepQuality', 'CholesterolHDL']

# Load the model
try:
    model = load('models/best_model.pkl')
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

@app.post('/prediction')
async def get_prediction(ADL: int, FunctionalAssessment: int, MMSE: int, BehavioralProblems: int,
                         MemoryComplaints: int, SleepQuality: int, CholesterolHDL: int):
    try:
        # Prepare the input data
        data = np.array([
            ADL,
            FunctionalAssessment,
            MMSE,
            BehavioralProblems,
            MemoryComplaints,
            SleepQuality,
            CholesterolHDL
        ], dtype=float).reshape(1, -1)  # Ensure data is float and reshape to match the expected input shape

        # Make the prediction
        prediction = model.predict(data)[0]

        return {"prediction": int(prediction)}  # Convert prediction to int if it's an integer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
