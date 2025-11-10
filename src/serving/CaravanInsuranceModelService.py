from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "xgb_classifier_model.joblib"

app = FastAPI()


class PredictRequest(BaseModel):
    features: Dict[str, float]


FEATURE_ORDER = [
    "MOSTYPE", "MAANTHUI", "MGEMOMV", "MGEMLEEF", "MOSHOOFD", "MGODRK", "MGODPR", "MGODOV", "MGODGE", "MRELGE",
    "MRELSA", "MRELOV", "MFALLEEN", "MFGEKIND", "MFWEKIND", "MOPLHOOG", "MOPLMIDD", "MOPLLAAG", "MBERHOOG", "MBERZELF",
    "MBERBOER", "MBERMIDD", "MBERARBG", "MBERARBO", "MSKA", "MSKB1", "MSKB2", "MSKC", "MSKD", "MHHUUR", "MHKOOP",
    "MAUT1",
    "MAUT2", "MAUT0", "MZFONDS", "MZPART", "MINKM30", "MINK3045", "MINK4575", "MINK7512", "MINK123M", "MINKGEM",
    "MKOOPKLA", "PWAPART", "PWABEDR", "PWALAND", "PPERSAUT", "PBESAUT", "PMOTSCO", "PVRAAUT", "PAANHANG", "PTRACTOR",
    "PWERKT", "PBROM", "PLEVEN", "PPERSONG", "PGEZONG", "PWAOREG", "PBRAND", "PZEILPL", "PPLEZIER", "PFIETS", "PINBOED",
    "PBYSTAND", "AWAPART", "AWABEDR", "AWALAND", "APERSAUT", "ABESAUT", "AMOTSCO", "AVRAAUT", "AAANHANG", "ATRACTOR",
    "AWERKT", "ABROM", "ALEVEN", "APERSONG", "AGEZONG", "AWAOREG", "ABRAND", "AZEILPL", "APLEZIER", "AFIETS", "AINBOED",
    "ABYSTAND", "86"
]

# Load the model from joblib file
model = joblib.load('pipeline.joblib')


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        # Check for missing features
        missing = [f for f in FEATURE_ORDER if f not in request.features]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

        # Arrange features in correct order
        ordered_values = [request.features[f] for f in FEATURE_ORDER]
        #np_input = np.array(ordered_values).reshape(1, -1)
        X = pd.DataFrame([ordered_values], columns=FEATURE_ORDER)
        # Prediction
        prediction = model.predict(X)
        return {"prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))