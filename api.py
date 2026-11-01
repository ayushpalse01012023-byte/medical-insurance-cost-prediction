"""
Vitalis AI - Medical Cost Prediction & Insurance Plan Recommendation API

Required files in the same folder:
    api.py
    medical_insurance_model.pkl
    feature_columns.pkl
    insurance_companies.csv

Run:
    uvicorn api:app --reload
"""

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict


# ============================================================
# CONFIGURATION
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vitalis_ai")

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "medical_insurance_model.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"
INSURANCE_CSV_PATH = BASE_DIR / "insurance_companies.csv"

NUM_RECOMMENDATIONS = 3

DEFAULT_ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "https://voluble-rugelach-ed09f4.netlify.app",
]

DISCLAIMER = (
    "SIMULATED DATA - ACADEMIC DEMONSTRATION ONLY. Insurance companies, "
    "plans and premiums in this dataset are fictional. Medical cost "
    "predictions are estimates and plan recommendations use simplified "
    "assumptions. They are not financial or insurance advice."
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


# ============================================================
# REQUEST MODEL
# ============================================================

class InsuranceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

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

    plan_type: Literal["Bronze", "Gold", "Platinum", "Silver"]
    network_tier: Literal["Bronze", "Gold", "Platinum", "Silver"]

    deductible: int = Field(..., ge=0)
    copay: int = Field(..., ge=0)
    policy_term_years: int = Field(..., ge=0)
    policy_changes_last_2yrs: int = Field(..., ge=0)

    provider_quality: float
    risk_score: float

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

    proc_imaging_count: int = Field(..., ge=0)
    proc_surgery_count: int = Field(..., ge=0)
    proc_physio_count: int = Field(..., ge=0)
    proc_consult_count: int = Field(..., ge=0)
    proc_lab_count: int = Field(..., ge=0)

    is_high_risk: int = Field(..., ge=0, le=1)
    had_major_procedure: int = Field(..., ge=0, le=1)


class PlanRecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    predicted_annual_medical_cost: float
    visits_last_year: Optional[int] = Field(default=4, ge=0)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

model = None
feature_columns = None
insurance_plans = None


# ============================================================
# MODEL PREPROCESSING
# ============================================================

def preprocess_input(payload: InsuranceInput):

    if feature_columns is None:
        raise RuntimeError("Feature columns are not loaded.")

    data = payload.model_dump()

    if data["alcohol_freq"] is None:
        data["alcohol_freq"] = "Occasional"

    categorical_columns = [
        "sex",
        "region",
        "urban_rural",
        "education",
        "marital_status",
        "employment_status",
        "smoker",
        "alcohol_freq",
        "plan_type",
        "network_tier",
    ]

    df = pd.DataFrame([data])

    # This assumes training used pandas get_dummies.
    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=False,
        dtype=int,
    )

    # Match the exact saved training feature order.
    df = df.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    return df


# ============================================================
# LOAD INSURANCE DATASET
# ============================================================

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
            f"Missing insurance CSV columns: {missing_columns}"
        )

    numeric_columns = [
        "annual_premium",
        "monthly_premium",
        "deductible",
        "copay",
        "coinsurance_percentage",
        "out_of_pocket_maximum",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=REQUIRED_PLAN_COLUMNS
    ).copy()

    if df.empty:
        raise ValueError("No valid insurance plans found.")

    for col in numeric_columns:
        if (df[col] < 0).any():
            raise ValueError(
                f"Negative values found in {col}."
            )

    if (df["coinsurance_percentage"] > 100).any():
        raise ValueError(
            "Coinsurance percentage cannot exceed 100."
        )

    if "data_status" not in df.columns:
        df["data_status"] = (
            "Simulated - Academic Demonstration Only"
        )

    return df.reset_index(drop=True)


# ============================================================
# INSURANCE PLAN COST CALCULATION
# ============================================================

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

    # Estimate annual copay expenses.
    copay_total = copay * visits

    # Remaining medical spending after estimated copays.
    remaining_cost = max(
        predicted_cost - copay_total,
        0,
    )

    # Apply deductible.
    deductible_payment = min(
        remaining_cost,
        deductible,
    )

    remaining_after_deductible = max(
        remaining_cost - deductible,
        0,
    )

    # Apply coinsurance.
    coinsurance_payment = (
        remaining_after_deductible * coinsurance
    )

    estimated_member_cost = (
        copay_total
        + deductible_payment
        + coinsurance_payment
    )

    # Academic assumption: all cost sharing counts toward
    # one annual out-of-pocket maximum.
    estimated_member_cost = min(
        estimated_member_cost,
        out_of_pocket_max,
    )

    estimated_total_cost = (
        annual_premium + estimated_member_cost
    )

    worst_case_cost = (
        annual_premium + out_of_pocket_max
    )

    return {
        "estimated_member_cost_sharing": estimated_member_cost,
        "estimated_total_annual_cost": estimated_total_cost,
        "worst_case_annual_cost": worst_case_cost,
    }


# ============================================================
# INSURANCE PLAN RANKING
# ============================================================

def rank_insurance_plans(predicted_cost, visits):

    df = insurance_plans.copy()

    calculations = df.apply(
        lambda row: calculate_plan_cost(
            row,
            predicted_cost,
            visits,
        ),
        axis=1,
    )

    calculated_df = pd.DataFrame(
        calculations.tolist(),
        index=df.index,
    )

    df = pd.concat(
        [df, calculated_df],
        axis=1,
    )

    # Lower total estimated annual cost gets a higher score.
    minimum_cost = df[
        "estimated_total_annual_cost"
    ].min()

    maximum_cost = df[
        "estimated_total_annual_cost"
    ].max()

    if maximum_cost == minimum_cost:
        df["cost_score"] = 100
    else:
        df["cost_score"] = (
            100
            * (maximum_cost - df["estimated_total_annual_cost"])
            / (maximum_cost - minimum_cost)
        )

    # Lower worst-case annual cost gets a higher score.
    minimum_worst = df[
        "worst_case_annual_cost"
    ].min()

    maximum_worst = df[
        "worst_case_annual_cost"
    ].max()

    if maximum_worst == minimum_worst:
        df["protection_score"] = 100
    else:
        df["protection_score"] = (
            100
            * (maximum_worst - df["worst_case_annual_cost"])
            / (maximum_worst - minimum_worst)
        )

    # Network tier scoring.
    network_scores = {
        "platinum": 100,
        "gold": 75,
        "silver": 50,
        "bronze": 25,
        "tier 1": 100,
        "tier 2": 75,
        "tier 3": 50,
    }

    df["network_score"] = (
        df["network_tier"]
        .astype(str)
        .str.lower()
        .map(network_scores)
    )

    # If network labels cannot be interpreted, do not
    # pretend they have a known ranking.
    if df["network_score"].isna().any():
        df["network_score"] = 0
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
            "estimated_total_annual_cost",
        ],
        ascending=[False, True],
    )

    return df.reset_index(drop=True)


# ============================================================
# STARTUP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model
    global feature_columns
    global insurance_plans

    required_files = [
        MODEL_PATH,
        FEATURE_COLUMNS_PATH,
        INSURANCE_CSV_PATH,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise RuntimeError(
            "Required files are missing:\n"
            + "\n".join(missing_files)
        )

    try:
        model = joblib.load(MODEL_PATH)

        feature_columns = joblib.load(
            FEATURE_COLUMNS_PATH
        )

        if not isinstance(
            feature_columns,
            (list, tuple, pd.Index),
        ):
            raise ValueError(
                "feature_columns.pkl must contain a list of feature names."
            )

        feature_columns = [
            str(col) for col in feature_columns
        ]

        if not feature_columns:
            raise ValueError(
                "Saved feature columns are empty."
            )

        if len(feature_columns) != len(set(feature_columns)):
            raise ValueError(
                "Duplicate feature names found."
            )

        insurance_plans = load_insurance_data()

        if not hasattr(model, "predict"):
            raise ValueError(
                "Loaded model does not support predict()."
            )

        logger.info("Vitalis AI backend loaded successfully.")
        logger.info(
            "Model features: %s",
            len(feature_columns),
        )
        logger.info(
            "Insurance plans: %s",
            len(insurance_plans),
        )

    except Exception as exc:
        logger.exception("Backend startup failed.")
        raise RuntimeError(
            f"Could not load Vitalis AI resources: {exc}"
        ) from exc

    yield

    logger.info("Vitalis AI backend stopped.")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Vitalis AI",
    description=(
        "Medical Cost Prediction and Insurance "
        "Plan Recommendation System"
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS CONFIGURATION
# ============================================================

allowed_origins = list(DEFAULT_ALLOWED_ORIGINS)

extra_origins = os.getenv(
    "FRONTEND_ORIGINS",
    "",
)

for origin in extra_origins.split(","):
    origin = origin.strip().rstrip("/")

    if origin and origin not in allowed_origins:
        allowed_origins.append(origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Vitalis AI API is running.",
        "documentation": "/docs",
        "health": "/health",
    }


# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================

@app.get("/health")
def health():

    ready = (
        model is not None
        and feature_columns is not None
        and insurance_plans is not None
    )

    return {
        "status": "ok" if ready else "not_ready",
        "model_loaded": model is not None,
        "feature_count": (
            len(feature_columns)
            if feature_columns is not None
            else 0
        ),
        "insurance_plans_loaded": (
            len(insurance_plans)
            if insurance_plans is not None
            else 0
        ),
    }


# ============================================================
# MEDICAL COST PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_medical_cost(payload: InsuranceInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not loaded.",
        )

    try:
        input_data = preprocess_input(payload)

        prediction = model.predict(input_data)

        predicted_cost = float(prediction[0])

        if not np.isfinite(predicted_cost):
            raise ValueError(
                "Model returned an invalid prediction."
            )

        return {
            "predicted_annual_medical_cost": round(
                predicted_cost,
                2,
            ),
            "currency": "USD",
            "message": (
                "Medical cost prediction generated successfully."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )


# ============================================================
# INSURANCE PLAN RECOMMENDATION ENDPOINT
# ============================================================

@app.post("/recommend-plans")
def recommend_insurance_plans(
    payload: PlanRecommendationRequest,
):

    if insurance_plans is None:
        raise HTTPException(
            status_code=503,
            detail="Insurance dataset is not loaded.",
        )

    try:
        predicted_cost = float(
            payload.predicted_annual_medical_cost
        )

        if not np.isfinite(predicted_cost):
            raise ValueError(
                "Predicted cost must be a finite number."
            )

        # Avoid using negative predicted spending for
        # insurance comparison.
        comparison_cost = max(predicted_cost, 0)

        visits = (
            payload.visits_last_year
            if payload.visits_last_year is not None
            else 4
        )

        ranked_plans = rank_insurance_plans(
            comparison_cost,
            visits,
        )

        recommended_plans = []

        top_plans = ranked_plans.head(
            NUM_RECOMMENDATIONS
        )

        for rank, (_, plan) in enumerate(
            top_plans.iterrows(),
            start=1,
        ):

            recommended_plans.append({
                "rank": rank,

                "company_name": plan["company_name"],
                "plan_name": plan["plan_name"],
                "plan_type": plan["plan_type"],

                "annual_premium": round(
                    float(plan["annual_premium"]),
                    2,
                ),

                "monthly_premium": round(
                    float(plan["monthly_premium"]),
                    2,
                ),

                "deductible": float(
                    plan["deductible"]
                ),

                "copay": float(
                    plan["copay"]
                ),

                "coinsurance_percentage": float(
                    plan["coinsurance_percentage"]
                ),

                "out_of_pocket_maximum": float(
                    plan["out_of_pocket_maximum"]
                ),

                "coverage_description": plan[
                    "coverage_description"
                ],

                "network_tier": plan["network_tier"],

                "estimated_member_cost_sharing": round(
                    float(
                        plan["estimated_member_cost_sharing"]
                    ),
                    2,
                ),

                "estimated_total_annual_cost": round(
                    float(
                        plan["estimated_total_annual_cost"]
                    ),
                    2,
                ),

                "worst_case_annual_cost": round(
                    float(
                        plan["worst_case_annual_cost"]
                    ),
                    2,
                ),

                "match_score": round(
                    float(plan["match_score"]),
                    2,
                ),

                "data_status": plan["data_status"],
            })

        return {
            "predicted_annual_medical_cost": round(
                predicted_cost,
                2,
            ),

            "currency": "USD",

            "data_status": (
                "Simulated - Academic Demonstration Only"
            ),

            "disclaimer": DISCLAIMER,

            "plans_evaluated": len(insurance_plans),

            "recommended_plans": recommended_plans,

            "assumptions": [
                "Medical cost prediction represents estimated annual medical spending.",
                "Annual premium is added to estimated member cost sharing.",
                "Copays are estimated using the number of annual visits.",
                "Deductible and coinsurance are applied using a simplified calculation.",
                "For this demonstration, copays, deductible and coinsurance are assumed to count toward one out-of-pocket maximum.",
                "Actual insurance policies may use different cost-sharing rules.",
                "All insurance plans and premiums are simulated for academic demonstration only.",
            ],
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Insurance recommendation failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Insurance recommendation failed: "
                + str(exc)
            ),
        )