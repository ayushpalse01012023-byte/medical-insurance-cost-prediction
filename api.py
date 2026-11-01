"""
Vitalis AI - Medical Cost Prediction and Insurance Plan Recommendation API

Required files:
    main.py
    medical_insurance_model.pkl
    feature_columns.pkl
    insurance_companies.csv

Run:
    uvicorn main:app --reload
"""

import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# =========================================================
# CONFIGURATION
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vitalis_ai")

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "medical_insurance_model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"
INSURANCE_CSV_PATH = BASE_DIR / "insurance_companies.csv"

ALCOHOL_FREQ_MODE_FALLBACK = "Occasional"
NUM_RECOMMENDATIONS = 3

DISCLAIMER = (
    "Insurance plans and company information are simulated "
    "for academic demonstration only. Cost calculations are "
    "estimates based on simplified assumptions and are not "
    "actual insurance quotations."
)

REQUIRED_PLAN_COLUMNS = [
    "company_name",
    "plan_name",
    "plan_type",
    "annual_premium",
    "monthly_premium",
    "deductible",
    "copay",
    "coinsurance_percentage",
    "out_of_pocket_maximum",
    "coverage_description",
    "network_tier",
]


# =========================================================
# GLOBAL VARIABLES
# =========================================================

model = None
feature_columns = None
insurance_plans = None


# =========================================================
# INPUT SCHEMA
# =========================================================

class InsuranceInput(BaseModel):

    # Demographics
    age: int = Field(..., ge=0, le=120)
    sex: Literal["Female", "Male", "Other"]
    region: Literal["Central", "East", "North", "South", "West"]
    urban_rural: Literal["Rural", "Suburban", "Urban"]
    income: float = Field(..., ge=0)
    education: Literal[
        "Bachelors", "Doctorate", "HS",
        "Masters", "No HS", "Some College"
    ]
    marital_status: Literal[
        "Divorced", "Married", "Single", "Widowed"
    ]
    employment_status: Literal[
        "Employed", "Retired", "Self-employed", "Unemployed"
    ]
    household_size: int = Field(..., ge=0)
    dependents: int = Field(..., ge=0)

    # Health and lifestyle
    bmi: float = Field(..., gt=0)
    smoker: Literal["Current", "Former", "Never"]
    alcohol_freq: Optional[
        Literal["Daily", "Occasional", "Weekly"]
    ] = None

    visits_last_year: int = Field(..., ge=0)
    hospitalizations_last_3yrs: int = Field(..., ge=0)
    days_hospitalized_last_3yrs: int = Field(..., ge=0)
    medication_count: int = Field(..., ge=0)

    systolic_bp: float = Field(..., gt=0)
    diastolic_bp: float = Field(..., gt=0)
    ldl: float = Field(..., ge=0)
    hba1c: float = Field(..., ge=0)

    # Insurance details used by the trained model
    plan_type: Literal["Bronze", "Gold", "Platinum", "Silver"]
    network_tier: Literal["Bronze", "Gold", "Platinum", "Silver"]

    deductible: int = Field(..., ge=0)
    copay: int = Field(..., ge=0)
    policy_term_years: int = Field(..., ge=0)
    policy_changes_last_2yrs: int = Field(..., ge=0)

    provider_quality: float
    risk_score: float

    # Chronic conditions
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

    # Procedure counts
    proc_imaging_count: int = Field(..., ge=0)
    proc_surgery_count: int = Field(..., ge=0)
    proc_physio_count: int = Field(..., ge=0)
    proc_consult_count: int = Field(..., ge=0)
    proc_lab_count: int = Field(..., ge=0)

    # Risk flags
    is_high_risk: int = Field(..., ge=0, le=1)
    had_major_procedure: int = Field(..., ge=0, le=1)

    class Config:
        extra = "forbid"


class PredictionResponse(BaseModel):
    predicted_annual_medical_cost: float


class PlanRecommendationRequest(BaseModel):
    predicted_annual_medical_cost: float = Field(..., ge=0)
    visits_last_year: int = Field(default=4, ge=0)


# =========================================================
# PREPROCESSING
# PRESERVES ORIGINAL TRAINING PREPROCESSING LOGIC
# =========================================================

def preprocess_input(payload: InsuranceInput) -> pd.DataFrame:

    if feature_columns is None:
        raise RuntimeError("Feature columns are not loaded.")

    input_dict = payload.dict()
    df = pd.DataFrame([input_dict])

    # Fill missing alcohol frequency using training mode.
    if df["alcohol_freq"].isnull().any():
        df["alcohol_freq"] = df["alcohol_freq"].fillna(
            ALCOHOL_FREQ_MODE_FALLBACK
        )

    # Same encoding method as original backend.
    df = pd.get_dummies(df, drop_first=True)

    # Convert boolean columns into integers.
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    # Match exact training feature names and order.
    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return df


# =========================================================
# LOAD INSURANCE DATASET
# =========================================================

def load_insurance_data():

    if not INSURANCE_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Insurance dataset not found: {INSURANCE_CSV_PATH}"
        )

    df = pd.read_csv(INSURANCE_CSV_PATH)

    missing_columns = [
        col for col in REQUIRED_PLAN_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing columns in insurance_companies.csv: "
            + ", ".join(missing_columns)
        )

    numeric_columns = [
        "annual_premium",
        "monthly_premium",
        "deductible",
        "copay",
        "coinsurance_percentage",
        "out_of_pocket_maximum"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_PLAN_COLUMNS).copy()

    if df.empty:
        raise ValueError("No valid insurance plans found.")

    for col in numeric_columns:
        if (df[col] < 0).any():
            raise ValueError(f"Invalid negative values in {col}.")

    if (df["coinsurance_percentage"] > 100).any():
        raise ValueError("Coinsurance cannot exceed 100 percent.")

    if "data_status" not in df.columns:
        df["data_status"] = "Simulated - Academic Demonstration Only"

    return df.reset_index(drop=True)


# =========================================================
# INSURANCE COST CALCULATION
# =========================================================

def calculate_plan_cost(plan, predicted_cost, visits):

    annual_premium = float(plan["annual_premium"])
    deductible = float(plan["deductible"])
    copay = float(plan["copay"])

    coinsurance = (
        float(plan["coinsurance_percentage"]) / 100
    )

    out_of_pocket_max = float(
        plan["out_of_pocket_maximum"]
    )

    # Estimated annual copay expenses.
    copay_total = copay * visits

    # Remaining medical expenses.
    remaining_cost = max(
        predicted_cost - copay_total,
        0
    )

    # Deductible payment.
    deductible_payment = min(
        remaining_cost,
        deductible
    )

    remaining_after_deductible = max(
        remaining_cost - deductible,
        0
    )

    # Coinsurance payment.
    coinsurance_payment = (
        remaining_after_deductible * coinsurance
    )

    # Estimated member expenses.
    member_cost = (
        copay_total
        + deductible_payment
        + coinsurance_payment
    )

    # Simplified academic assumption:
    # all cost-sharing expenses count toward one OOP maximum.
    member_cost = min(
        member_cost,
        out_of_pocket_max
    )

    total_annual_cost = annual_premium + member_cost

    worst_case_cost = (
        annual_premium + out_of_pocket_max
    )

    return {
        "estimated_member_cost_sharing": member_cost,
        "estimated_total_annual_cost": total_annual_cost,
        "worst_case_annual_cost": worst_case_cost
    }


# =========================================================
# RANK INSURANCE PLANS
# =========================================================

def rank_insurance_plans(predicted_cost, visits):

    df = insurance_plans.copy()

    calculations = df.apply(
        lambda row: calculate_plan_cost(
            row,
            predicted_cost,
            visits
        ),
        axis=1
    )

    calculated_df = pd.DataFrame(
        calculations.tolist(),
        index=df.index
    )

    df = pd.concat([df, calculated_df], axis=1)

    # Lower total estimated cost = higher score.
    min_cost = df["estimated_total_annual_cost"].min()
    max_cost = df["estimated_total_annual_cost"].max()

    if max_cost == min_cost:
        df["cost_score"] = 100
    else:
        df["cost_score"] = (
            100 * (max_cost - df["estimated_total_annual_cost"])
            / (max_cost - min_cost)
        )

    # Lower worst-case cost = higher protection score.
    min_worst = df["worst_case_annual_cost"].min()
    max_worst = df["worst_case_annual_cost"].max()

    if max_worst == min_worst:
        df["protection_score"] = 100
    else:
        df["protection_score"] = (
            100 * (max_worst - df["worst_case_annual_cost"])
            / (max_worst - min_worst)
        )

    # Network tier score.
    network_scores = {
        "platinum": 100,
        "gold": 75,
        "silver": 50,
        "bronze": 25,
        "tier 1": 100,
        "tier 2": 75,
        "tier 3": 50
    }

    df["network_score"] = (
        df["network_tier"]
        .astype(str)
        .str.lower()
        .map(network_scores)
    )

    if df["network_score"].isna().any():
        # Do not assume an unknown network tier ranking.
        df["match_score"] = (
            df["cost_score"] * 0.70
            + df["protection_score"] * 0.30
        )
    else:
        df["match_score"] = (
            df["cost_score"] * 0.60
            + df["protection_score"] * 0.25
            + df["network_score"] * 0.15
        )

    df = df.sort_values(
        by=[
            "match_score",
            "estimated_total_annual_cost"
        ],
        ascending=[False, True]
    )

    return df.reset_index(drop=True)


# =========================================================
# APPLICATION STARTUP
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model, feature_columns, insurance_plans

    try:
        logger.info("Loading CatBoost model...")

        model = joblib.load(MODEL_PATH)

        logger.info("Loading feature columns...")

        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

        if not isinstance(feature_columns, (list, tuple, pd.Index)):
            raise ValueError(
                "feature_columns.pkl must contain feature names."
            )

        feature_columns = list(feature_columns)

        if not feature_columns:
            raise ValueError("Feature columns are empty.")

        logger.info("Loading insurance companies dataset...")

        insurance_plans = load_insurance_data()

        logger.info("Vitalis AI backend loaded successfully.")

        logger.info("Model features: %s", len(feature_columns))
        logger.info("Insurance plans: %s", len(insurance_plans))

    except Exception as exc:
        logger.exception("Backend startup failed.")
        raise RuntimeError(str(exc)) from exc

    yield

    logger.info("Vitalis AI backend shut down.")


# =========================================================
# FASTAPI INITIALIZATION
# =========================================================

app = FastAPI(
    title="Vitalis AI",
    description=(
        "Medical Cost Prediction and Insurance "
        "Plan Recommendation System"
    ),
    version="2.0.0",
    lifespan=lifespan
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://voluble-rugelach-ed09f4.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def read_root():

    return {
        "message": "Welcome to Vitalis AI API.",
        "prediction_endpoint": "/predict",
        "recommendation_endpoint": "/recommend-plans",
        "health_endpoint": "/health",
        "documentation": "/docs"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "feature_count": (
            len(feature_columns)
            if feature_columns is not None else 0
        ),
        "insurance_plans_loaded": (
            len(insurance_plans)
            if insurance_plans is not None else 0
        )
    }


# =========================================================
# MEDICAL COST PREDICTION
# =========================================================

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: InsuranceInput):

    if model is None or feature_columns is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not loaded."
        )

    try:
        processed_df = preprocess_input(payload)

        prediction = model.predict(processed_df)

        predicted_value = float(prediction[0])

        if not np.isfinite(predicted_value):
            raise ValueError("Model returned an invalid prediction.")

        return PredictionResponse(
            predicted_annual_medical_cost=round(
                predicted_value, 2
            )
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )


# =========================================================
# INSURANCE PLAN RECOMMENDATIONS
# =========================================================

@app.post("/recommend-plans")
def recommend_plans(payload: PlanRecommendationRequest):

    if insurance_plans is None:
        raise HTTPException(
            status_code=503,
            detail="Insurance dataset is not loaded."
        )

    try:
        predicted_cost = float(
            payload.predicted_annual_medical_cost
        )

        if not np.isfinite(predicted_cost):
            raise ValueError("Invalid predicted medical cost.")

        visits = payload.visits_last_year

        ranked = rank_insurance_plans(
            predicted_cost,
            visits
        )

        top_plans = ranked.head(NUM_RECOMMENDATIONS)

        recommendations = []

        for rank, (_, plan) in enumerate(
            top_plans.iterrows(),
            start=1
        ):

            recommendations.append({
                "rank": rank,
                "company_name": str(plan["company_name"]),
                "plan_name": str(plan["plan_name"]),
                "plan_type": str(plan["plan_type"]),

                "annual_premium": round(
                    float(plan["annual_premium"]), 2
                ),

                "monthly_premium": round(
                    float(plan["monthly_premium"]), 2
                ),

                "deductible": float(plan["deductible"]),
                "copay": float(plan["copay"]),

                "coinsurance_percentage": float(
                    plan["coinsurance_percentage"]
                ),

                "out_of_pocket_maximum": float(
                    plan["out_of_pocket_maximum"]
                ),

                "coverage_description": str(
                    plan["coverage_description"]
                ),

                "network_tier": str(plan["network_tier"]),

                "estimated_member_cost_sharing": round(
                    float(plan["estimated_member_cost_sharing"]),
                    2
                ),

                "estimated_total_annual_cost": round(
                    float(plan["estimated_total_annual_cost"]),
                    2
                ),

                "worst_case_annual_cost": round(
                    float(plan["worst_case_annual_cost"]),
                    2
                ),

                "match_score": round(
                    float(plan["match_score"]), 2
                ),

                "data_status": str(plan["data_status"])
            })

        return {
            "predicted_annual_medical_cost": round(
                predicted_cost, 2
            ),

            "currency": "USD",

            "plans_evaluated": len(insurance_plans),

            "recommended_plans": recommendations,

            "data_status": "Simulated - Academic Demonstration Only",

            "disclaimer": DISCLAIMER,

            "calculation_assumptions": [
                "Annual premium is added to estimated member cost sharing.",
                "Copay expenses are estimated using annual doctor visits.",
                "Deductible and coinsurance use a simplified calculation.",
                "All cost-sharing expenses are assumed to count toward one out-of-pocket maximum.",
                "Actual insurance policy rules may differ."
            ]
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Insurance recommendation failed.")

        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(exc)}"
        )


# =========================================================
# END OF VITALIS AI BACKEND
# =========================================================