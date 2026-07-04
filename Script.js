// =========================================================
// VITALIS — Prediction Client
// =========================================================

const API_URL = "http://127.0.0.1:8000/predict";

const form = document.getElementById("predict-form");
const submitBtn = document.getElementById("submit-btn");
const errorNote = document.getElementById("form-error");

const monitorCard = document.getElementById("monitor-card");
const monitorId = document.getElementById("monitor-id");
const costValueEl = document.getElementById("cost-value");
const riskChip = document.getElementById("risk-chip");
const resultEcgPath = document.getElementById("result-ecg-path");

// Field name lists, kept exactly as required by the FastAPI Pydantic model.
const NUMBER_FIELDS = [
  "age", "income", "household_size", "dependents",
  "bmi", "visits_last_year", "hospitalizations_last_3yrs",
  "days_hospitalized_last_3yrs", "medication_count",
  "systolic_bp", "diastolic_bp", "ldl", "hba1c",
  "deductible", "copay", "policy_term_years",
  "policy_changes_last_2yrs", "provider_quality", "risk_score",
  "chronic_count",
  "proc_imaging_count", "proc_surgery_count", "proc_physio_count",
  "proc_consult_count", "proc_lab_count"
];

const SELECT_FIELDS = [
  "sex", "region", "urban_rural", "education", "marital_status",
  "employment_status", "smoker", "alcohol_freq", "plan_type", "network_tier"
];

const BOOLEAN_FIELDS = [
  "hypertension", "diabetes", "asthma", "copd",
  "cardiovascular_disease", "cancer_history", "kidney_disease",
  "liver_disease", "arthritis", "mental_health",
  "is_high_risk", "had_major_procedure"
];

/**
 * Builds the JSON payload sent to the FastAPI /predict endpoint.
 * Field names below are character-for-character identical to the
 * backend's Pydantic model — do not rename, camelCase, or omit any key.
 */
function buildPayload() {
  const payload = {};

  NUMBER_FIELDS.forEach((field) => {
    const el = document.getElementById(field);
    payload[field] = el.value === "" ? null : Number(el.value);
  });

  SELECT_FIELDS.forEach((field) => {
    const el = document.getElementById(field);
    payload[field] = el.value;
  });

  BOOLEAN_FIELDS.forEach((field) => {
    const el = document.getElementById(field);
    payload[field] = el.checked ? 1 : 0;
  });

  return payload;
}

function validatePayload(payload) {
  const missing = [];

  NUMBER_FIELDS.forEach((field) => {
    if (payload[field] === null || Number.isNaN(payload[field])) missing.push(field);
  });
  SELECT_FIELDS.forEach((field) => {
    if (!payload[field]) missing.push(field);
  });

  return missing;
}

function setLoading(isLoading) {
  submitBtn.classList.toggle("is-loading", isLoading);
  submitBtn.disabled = isLoading;
}

function showFormError(message) {
  errorNote.textContent = message;
}

function riskLabel(payload, predictedCost) {
  if (payload.is_high_risk === 1) return { text: "High Risk", cls: "risk-high" };
  if (predictedCost >= 15000) return { text: "Elevated Risk", cls: "risk-elevated" };
  return { text: "Low Risk", cls: "risk-low" };
}

function animateCount(target) {
  const duration = 900;
  const start = performance.now();
  const from = 0;

  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = Math.round(from + (target - from) * eased);
    costValueEl.textContent = value.toLocaleString("en-US");
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function animateResultEcg(intensity) {
  // intensity 0..1 drives how "spiked" the readout line looks
  const amp = 6 + intensity * 18;
  const w = 400, midY = 30;
  let d = `M0 ${midY} `;
  const segments = 10;
  for (let i = 0; i < segments; i++) {
    const x = (w / segments) * i;
    const nextX = (w / segments) * (i + 1);
    const spikeUp = midY - amp * (0.4 + Math.random() * 0.6);
    const spikeDown = midY + amp * (0.3 + Math.random() * 0.5);
    d += `L ${x + 8} ${midY} L ${x + 14} ${spikeUp} L ${x + 20} ${spikeDown} L ${nextX} ${midY} `;
  }
  resultEcgPath.setAttribute("d", d);
}

function extractCost(data) {
  // Accept a range of reasonable response shapes without guessing field
  // names on the request side — this only reads the response.
  if (typeof data === "number") return data;
  if (data && typeof data === "object") {
    const candidates = [
      "prediction", "predicted_cost", "predicted_annual_cost",
      "cost", "annual_medical_cost", "result", "value"
    ];
    for (const key of candidates) {
      if (key in data && typeof data[key] === "number") return data[key];
    }
    // fall back: first numeric value found in the object
    const firstNumeric = Object.values(data).find((v) => typeof v === "number");
    if (firstNumeric !== undefined) return firstNumeric;
  }
  return null;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  showFormError("");

  const payload = buildPayload();
  const missing = validatePayload(payload);

  if (missing.length > 0) {
    showFormError(`Please complete: ${missing.join(", ")}`);
    document.getElementById(missing[0]).focus();
    return;
  }

  setLoading(true);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      let detail = `Request failed with status ${response.status}.`;
      try {
        const errBody = await response.json();
        if (errBody && errBody.detail) {
          detail = typeof errBody.detail === "string"
            ? errBody.detail
            : JSON.stringify(errBody.detail);
        }
      } catch (_) {
        // response had no JSON body — keep the generic message
      }
      throw new Error(detail);
    }

    const data = await response.json();
    const cost = extractCost(data);

    if (cost === null) {
      throw new Error("The prediction service returned an unexpected response shape.");
    }

    monitorCard.classList.add("is-active");
    monitorId.textContent = `CASE #${Math.floor(100000 + Math.random() * 900000)}`;
    animateCount(Math.round(cost));

    const risk = riskLabel(payload, cost);
    riskChip.textContent = risk.text;
    riskChip.className = `risk-chip ${risk.cls}`;

    animateResultEcg(Math.min(cost / 25000, 1));

    document.getElementById("result").scrollIntoView({ behavior: "smooth", block: "center" });

  } catch (err) {
    showFormError(
      err.message.includes("Failed to fetch")
        ? "Could not reach the prediction service at 127.0.0.1:8000. Confirm the FastAPI backend is running."
        : err.message
    );
  } finally {
    setLoading(false);
  }
});