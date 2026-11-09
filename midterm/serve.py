import pickle

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Medical Insurance Cost Prediction")

with open("model.pkl", "rb") as f:
    artifacts = pickle.load(f)

dv = artifacts["dv"]
model = artifacts["model"]


class Person(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(person: Person):
    features = person.model_dump()
    X = dv.transform([features])
    charges = float(model.predict(X)[0])
    return {"charges": round(charges, 2)}
