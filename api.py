"""
main.py

FastAPI backend for the "Medical Insurance Cost Prediction" project.

Serves predictions from a trained CatBoostRegressor model that was
trained to predict `annual_medical_cost` from a set of applicant /
policy / health features.

Run with:
    uvicorn main:app --reload
"""

# =========================================================
# IMPORTS
# =========================================================

import logging
from contextlib import asynccontextmanager
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# =========================================================
# LOGGING SETUP
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("insurance_prediction_api")


# =========================================================
# CONSTANTS
# =========================================================

MODEL_PATH = "medical_insurance_model.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"

# Mode value used to fill missing `alcohol_freq` entries, taken directly
# from the training dataset via df["alcohol_freq"].mode()[0].
ALCOHOL_FREQ_MODE_FALLBACK = "Occasional"


# =========================================================
# GLOBAL OBJECTS (populated once at startup)
# =========================================================

# These are populated in the lifespan startup handler below and reused
# across every request, so the model and feature list are loaded from
# disk only ONCE for the lifetime of the application process.
model = None
feature_columns = None


# =========================================================
# APPLICATION LIFESPAN (load model + feature columns once)
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the application starts up, and once when it shuts
    down. Used here to load the trained model and the feature-column
    list into memory a single time, instead of reloading them on
    every request (which would be slow and wasteful).
    """
    global model, feature_columns

    try:
        logger.info("Loading model from '%s'...", MODEL_PATH)
        model = joblib.load(MODEL_PATH)

        logger.info("Loading feature columns from '%s'...", FEATURE_COLUMNS_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

        logger.info("Model and feature columns loaded successfully.")
    except FileNotFoundError as exc:
        # Fail loudly at startup if the required artifacts are missing,
        # rather than letting the app start in a broken state.
        logger.error("Required model artifact not found: %s", exc)
        raise
    except Exception as exc:
        logger.error("Unexpected error while loading model artifacts: %s", exc)
        raise

    yield  # Application runs while suspended here

    # Optional cleanup on shutdown (nothing required for this project)
    logger.info("Shutting down Medical Insurance Cost Prediction API.")


# =========================================================
# FASTAPI APP INITIALIZATION
# =========================================================

app = FastAPI(
    title="Medical Insurance Cost Prediction API",
    description="Predicts annual medical cost using a trained CatBoostRegressor model.",
    version="1.0.0",
    lifespan=lifespan,
)


# =========================================================
# REQUEST SCHEMA (Pydantic model)
# =========================================================
#
# This defines every field the client is allowed to send. Fields that
# were dropped before training (person_id, annual_medical_cost,
# annual_premium, monthly_premium, total_claims_paid,
# avg_claim_amount, claims_count) are intentionally NOT included here,
# so FastAPI/Pydantic will reject any request that tries to send them
# (extra = "forbid" below enforces this strictly).

class InsuranceInput(BaseModel):
    # --- Demographics ---
    age: int = Field(..., ge=0, le=120, description="Applicant age in years")
    sex: Literal["Female", "Male", "Other"]
    region: Literal["Central", "East", "North", "South", "West"]
    urban_rural: Literal["Rural", "Suburban", "Urban"]
    income: float = Field(..., ge=0)
    education: Literal["Bachelors", "Doctorate", "HS", "Masters", "No HS", "Some College"]
    marital_status: Literal["Divorced", "Married", "Single", "Widowed"]
    employment_status: Literal["Employed", "Retired", "Self-employed", "Unemployed"]
    household_size: int = Field(..., ge=0)
    dependents: int = Field(..., ge=0)

    # --- Health / lifestyle ---
    bmi: float = Field(..., gt=0)
    smoker: Literal["Current", "Former", "Never"]
    # Optional so that missing values can go through the mode-fill step
    # described in the preprocessing pipeline.
    alcohol_freq: Optional[Literal["Daily", "Occasional", "Weekly"]] = None
    visits_last_year: int = Field(..., ge=0)
    hospitalizations_last_3yrs: int = Field(..., ge=0)
    days_hospitalized_last_3yrs: int = Field(..., ge=0)
    medication_count: int = Field(..., ge=0)
    systolic_bp: float = Field(..., gt=0)
    diastolic_bp: float = Field(..., gt=0)
    ldl: float = Field(..., ge=0)
    hba1c: float = Field(..., ge=0)

    # --- Plan / policy details ---
    plan_type: Literal["Bronze", "Gold", "Platinum", "Silver"]
    network_tier: Literal["Bronze", "Gold", "Platinum", "Silver"]
    deductible: int = Field(..., ge=0)
    copay: int = Field(..., ge=0)
    policy_term_years: int = Field(..., ge=0)
    policy_changes_last_2yrs: int = Field(..., ge=0)
    provider_quality: float
    risk_score: float

    # --- Chronic conditions (binary flags: 0 or 1) ---
    chronic_count: int = Field(..., ge=0)
    hypertension: int = Field(..., ge=0, le=1)
    diabetes: int = Field(..., ge=0, le=1)
    asthma: int = Field(..., ge=0, le=1)
    copd: int = Field(..., ge=0, le=1)
    cardiovascular_disease: int = Field(..., ge=0, le=1)
    cancer_history: int = Field(..., ge=0, le=1)
    kidney_disease: int = Field(..., ge=0, le=1)
    liver_disease: int = Field(..., ge=0, le=1)
    arthritis: int = Field(..., ge=0, le=1)
    mental_health: int = Field(..., ge=0, le=1)

    # --- Procedure counts ---
    proc_imaging_count: int = Field(..., ge=0)
    proc_surgery_count: int = Field(..., ge=0)
    proc_physio_count: int = Field(..., ge=0)
    proc_consult_count: int = Field(..., ge=0)
    proc_lab_count: int = Field(..., ge=0)

    # --- Derived risk flags ---
    is_high_risk: int = Field(..., ge=0, le=1)
    had_major_procedure: int = Field(..., ge=0, le=1)

    class Config:
        # Reject any field not explicitly defined above (e.g. person_id,
        # annual_medical_cost, annual_premium, etc.), preventing target
        # leakage or accidental misuse of the API.
        extra = "forbid"

        json_schema_extra = {
            "example": {
                "age": 45,
                "sex": "Male",
                "region": "North",
                "urban_rural": "Urban",
                "income": 65000.0,
                "education": "Bachelors",
                "marital_status": "Married",
                "employment_status": "Employed",
                "household_size": 4,
                "dependents": 2,
                "bmi": 27.5,
                "smoker": "Never",
                "alcohol_freq": "Occasional",
                "visits_last_year": 3,
                "hospitalizations_last_3yrs": 0,
                "days_hospitalized_last_3yrs": 0,
                "medication_count": 1,
                "systolic_bp": 120.0,
                "diastolic_bp": 80.0,
                "ldl": 110.0,
                "hba1c": 5.4,
                "plan_type": "Gold",
                "network_tier": "Gold",
                "deductible": 500,
                "copay": 20,
                "policy_term_years": 3,
                "policy_changes_last_2yrs": 0,
                "provider_quality": 4.2,
                "risk_score": 0.35,
                "chronic_count": 0,
                "hypertension": 0,
                "diabetes": 0,
                "asthma": 0,
                "copd": 0,
                "cardiovascular_disease": 0,
                "cancer_history": 0,
                "kidney_disease": 0,
                "liver_disease": 0,
                "arthritis": 0,
                "mental_health": 0,
                "proc_imaging_count": 1,
                "proc_surgery_count": 0,
                "proc_physio_count": 0,
                "proc_consult_count": 2,
                "proc_lab_count": 3,
                "is_high_risk": 0,
                "had_major_procedure": 0,
            }
        }


# =========================================================
# RESPONSE SCHEMA
# =========================================================

class PredictionResponse(BaseModel):
    predicted_annual_medical_cost: float


# =========================================================
# PREPROCESSING FUNCTION
# =========================================================
#
# This function replicates, step-by-step, the exact preprocessing
# pipeline that was used before training the CatBoostRegressor model.

def preprocess_input(payload: InsuranceInput) -> pd.DataFrame:
    """
    Convert a validated InsuranceInput request into a model-ready
    pandas DataFrame using the same preprocessing steps applied
    during training.
    """

    # ---------------------------------------------------
    # Step 1: Create a pandas DataFrame from the incoming request
    # ---------------------------------------------------
    input_dict = payload.model_dump()
    df = pd.DataFrame([input_dict])

    # ---------------------------------------------------
    # Step 2: Fill missing values in alcohol_freq using the mode
    # ---------------------------------------------------
    if df["alcohol_freq"].isnull().any():
        df["alcohol_freq"] = df["alcohol_freq"].fillna(ALCOHOL_FREQ_MODE_FALLBACK)

    # ---------------------------------------------------
    # Step 3: One-hot encode categorical columns (drop_first=True),
    # exactly as done during training.
    # ---------------------------------------------------
    df = pd.get_dummies(df, drop_first=True)

    # ---------------------------------------------------
    # Step 4: Convert boolean columns into integers
    # (get_dummies can produce bool dtype columns depending on
    # the pandas version; the model expects numeric input).
    # ---------------------------------------------------
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # ---------------------------------------------------
    # Step 5: Reindex to match the exact column order used during
    # training. Any column present at training time but missing here
    # (because this single request didn't produce that dummy category)
    # is added back with a value of 0. Any unexpected column is dropped.
    # ---------------------------------------------------
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


# =========================================================
# ROUTES
# =========================================================

@app.get("/")
def read_root():
    """
    Simple welcome/health-check endpoint.
    """
    return {
        "message": "Welcome to the Medical Insurance Cost Prediction API. "
                   "Use POST /predict to get a prediction."
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: InsuranceInput):
    """
    Accepts applicant/policy/health data, preprocesses it exactly as
    done during training, and returns the predicted annual medical
    cost from the trained CatBoostRegressor model.
    """

    # Guard clause: make sure the model actually loaded at startup.
    if model is None or feature_columns is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please try again shortly.",
        )

    try:
        # Preprocess the incoming request into a model-ready DataFrame.
        processed_df = preprocess_input(payload)
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        raise HTTPException(
            status_code=400,
            detail=f"Failed to preprocess input data: {exc}",
        )

    try:
        # Run inference using the trained CatBoostRegressor.
        prediction = model.predict(processed_df)
        predicted_value = float(prediction[0])
    except Exception as exc:
        logger.error("Model prediction failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Model failed to generate a prediction: {exc}",
        )

    return PredictionResponse(
        predicted_annual_medical_cost=round(predicted_value, 2)
    )


# =========================================================
# HOW TO RUN
# =========================================================
#
#   uvicorn main:app --reload
#
# Make sure 'medical_insurance_model.pkl' and 'feature_columns.pkl'
# are located in the same directory as this file (or update
# MODEL_PATH / FEATURE_COLUMNS_PATH above to point to their location).