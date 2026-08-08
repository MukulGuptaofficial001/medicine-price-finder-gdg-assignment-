import os
import re
import joblib
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from scipy.sparse import hstack
from typing import List

model = None
le = None
tfidf = None
df = None


def clean_comp(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'(\d+)\s*(mg|ml|mcg|iu|g)\b', r'\1\2', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, le, tfidf, df
    model = joblib.load(os.path.join("model", "model.joblib"))
    le = joblib.load(os.path.join("model", "encoder.joblib"))
    tfidf = joblib.load(os.path.join("model", "tfidf.joblib"))
    df = pd.read_csv(os.path.join("model", "model_data.csv"))
    yield


app = FastAPI(title="Medicine Price API", lifespan=lifespan)


class PredictResponse(BaseModel):
    medicine_name: str
    predicted_price: float
    manufacturer: str
    composition: str


class Alternative(BaseModel):
    medicine_name: str
    manufacturer: str
    price: float
    composition: str


class AlternativesResponse(BaseModel):
    medicine_name: str
    composition: str
    alternatives: List[Alternative]


def get_medicine_row(m_name: str):
    mask = df["medicine_name"].str.lower() == m_name.lower()
    matches = df[mask]
    if matches.empty:
        mask2 = df["medicine_name"].str.lower().str.contains(m_name.lower(), na=False)
        matches = df[mask2]
    if matches.empty:
        return None
    return matches.iloc[0]


def predict_for_row(row):
    comp_vec = tfidf.transform([row["cleaned_composition"]])
    manuf_str = str(row["manufacturer"])
    if manuf_str in le.classes_:
        manuf_enc = le.transform([manuf_str])[0]
    else:
        manuf_enc = 0
    manuf_arr = np.array([[manuf_enc]])
    X = hstack([comp_vec, manuf_arr])
    return float(model.predict(X)[0])


@app.get("/predict-price", response_model=PredictResponse)
def predict_price(medicine_name: str = Query(..., min_length=1)):
    row = get_medicine_row(medicine_name)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Medicine '{medicine_name}' not found in dataset")
    predicted = predict_for_row(row)
    return PredictResponse(
        medicine_name=str(row["medicine_name"]),
        predicted_price=round(predicted, 2),
        manufacturer=str(row["manufacturer"]),
        composition=str(row["cleaned_composition"])
    )


@app.get("/alternatives", response_model=AlternativesResponse)
def get_alternatives(medicine_name: str = Query(..., min_length=1)):
    row = get_medicine_row(medicine_name)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Medicine '{medicine_name}' not found in dataset")
    target_comp = row["cleaned_composition"]
    same_comp = df[df["cleaned_composition"] == target_comp].copy()
    same_comp = same_comp[same_comp["medicine_name"].str.lower() != str(row["medicine_name"]).lower()]
    same_comp = same_comp.sort_values("price").head(5)
    alts = []
    for _, r in same_comp.iterrows():
        alts.append(Alternative(
            medicine_name=str(r["medicine_name"]),
            manufacturer=str(r["manufacturer"]),
            price=float(r["price"]),
            composition=str(r["cleaned_composition"])
        ))
    return AlternativesResponse(
        medicine_name=str(row["medicine_name"]),
        composition=str(row["cleaned_composition"]),
        alternatives=alts
    )


@app.get("/")
def root():
    return {"status": "ok", "message": "Medicine Price API is running"}
