
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Customer Churn Prediction API")

model = joblib.load("customer_churn_model.pkl")


class CustomerData(BaseModel):
    tenure_months: int
    support_tickets: int
    monthly_spend_inr: float
    last_login_days: int
    plan_type: str


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running!"}


@app.post("/predict")
def predict_churn(customer: CustomerData):
    data = pd.DataFrame([customer.model_dump()])
    prediction = int(model.predict(data)[0])

    return {
        "churn_prediction": prediction,
        "result": "Likely to Churn" if prediction == 1 else "Not Likely to Churn"
    }
