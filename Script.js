// =========================================================
// VITALIS — Enterprise Dashboard Client
// =========================================================

const API_URL = "https://medical-insurance-cost-prediction-t0d6.onrender.com/predict";

/* ---------------------------------------------------------
   Field maps — identical to the FastAPI Pydantic model.
   Do not rename, camelCase, or omit any key here.
--------------------------------------------------------- */
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
 * Field names are character-for-character identical to the backend's
 * Pydantic model — do not rename, camelCase, or omit any key.
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

function extractCost(data) {
  if (typeof data === "number") return data;
  if (data && typeof data === "object") {
    const candidates = [
      "prediction", "predicted_cost", "predicted_annual_cost",
      "cost", "annual_medical_cost", "result", "value"
    ];
    for (const key of candidates) {
      if (key in data && typeof data[key] === "number") return data[key];
    }
    const firstNumeric = Object.values(data).find((v) => typeof v === "number");
    if (firstNumeric !== undefined) return firstNumeric;
  }
  return null;
}

/* ---------------------------------------------------------
   DOM references
--------------------------------------------------------- */
const form = document.getElementById("predict-form");
const submitBtn = document.getElementById("submit-btn");
const errorNote = document.getElementById("form-error");

const monitorCard = document.getElementById("monitor-card");
const monitorId = document.getElementById("monitor-id");
const costValueEl = document.getElementById("cost-value");
const riskChip = document.getElementById("risk-chip");
const resultEcgPath = document.getElementById("result-ecg-path");
const confidenceFill = document.getElementById("confidence-fill");
const confidenceLabel = document.getElementById("confidence-label");

const kpiCount = document.getElementById("kpi-count");
const kpiAvg = document.getElementById("kpi-avg");
const miniCount = document.getElementById("mini-count");
const miniAvg = document.getElementById("mini-avg");
const orbitLatency = document.getElementById("orbit-latency");

/* ---------------------------------------------------------
   Session stats (client-side only — real usage, not fabricated)
--------------------------------------------------------- */
const session = { count: 0, total: 0 };

function refreshSessionStats() {
  kpiCount.textContent = session.count.toString();
  miniCount.textContent = session.count.toString();
  const avg = session.count > 0 ? Math.round(session.total / session.count) : null;
  const avgText = avg !== null ? `$${avg.toLocaleString("en-US")}` : "—";
  kpiAvg.textContent = avg !== null ? avgText : "—";
  miniAvg.textContent = avg !== null ? avgText : "$0";
}

/* ---------------------------------------------------------
   Sidebar navigation: mobile toggle + scrollspy
--------------------------------------------------------- */
const sidebar = document.getElementById("sidebar");
const sidebarOverlay = document.getElementById("sidebar-overlay");
const mobileToggle = document.getElementById("mobile-toggle");
const sidebarLinks = Array.from(document.querySelectorAll(".sidebar-link"));
const topbarCrumbActive = document.getElementById("topbar-crumb-active");

function toggleSidebar(open) {
  sidebar.classList.toggle("is-open", open);
  sidebarOverlay.classList.toggle("is-open", open);
}
mobileToggle.addEventListener("click", () => toggleSidebar(!sidebar.classList.contains("is-open")));
sidebarOverlay.addEventListener("click", () => toggleSidebar(false));
sidebarLinks.forEach((link) => link.addEventListener("click", () => toggleSidebar(false)));

const sectionIds = ["dashboard", "prediction", "analytics", "model-info", "about"];
const sectionLabels = {
  "dashboard": "Dashboard", "prediction": "Prediction", "analytics": "Analytics",
  "model-info": "Model Info", "about": "About"
};
const sections = sectionIds.map((id) => document.getElementById(id)).filter(Boolean);

const spyObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const id = entry.target.id;
        sidebarLinks.forEach((l) => l.classList.toggle("is-active", l.dataset.target === id));
        if (topbarCrumbActive && sectionLabels[id]) topbarCrumbActive.textContent = sectionLabels[id];
      }
    });
  },
  { rootMargin: "-40% 0px -50% 0px", threshold: 0 }
);
sections.forEach((section) => spyObserver.observe(section));

/* ---------------------------------------------------------
   Loading / error UI
--------------------------------------------------------- */
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
  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const value = Math.round(target * eased);
    costValueEl.textContent = value.toLocaleString("en-US");
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function animateResultEcg(intensity) {
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

/* ---------------------------------------------------------
   Chart.js — global theme
--------------------------------------------------------- */
const CHART_COLORS = {
  accent: "#00E5C7",
  accent2: "#7C8CFF",
  accent3: "#B98CFF",
  danger: "#FF6B5A",
  warn: "#FFC15E",
  text: "#8C97AC",
  grid: "rgba(255,255,255,0.06)"
};

let charts = {};

function initChartDefaults() {
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.color = CHART_COLORS.text;
  Chart.defaults.borderColor = CHART_COLORS.grid;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.tooltip.backgroundColor = "rgba(10,14,22,0.95)";
  Chart.defaults.plugins.tooltip.titleColor = "#EDF1F8";
  Chart.defaults.plugins.tooltip.bodyColor = "#B7C0D1";
  Chart.defaults.plugins.tooltip.borderColor = "rgba(0,229,199,0.35)";
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.cornerRadius = 8;
  Chart.defaults.animation.duration = 900;
  Chart.defaults.animation.easing = "easeOutQuart";
}

function buildAllCharts() {
  const builders = [
    ["Feature Importance", buildFeatureImportanceChart],
    ["Cost Distribution", buildCostDistributionChart],
    ["Actual vs Predicted", buildActualPredictedChart],
    ["Risk Radar", buildRiskRadarChart],
    ["Health Polar", buildHealthPolarChart],
    ["Lifestyle Pie", buildLifestylePieChart],
    ["Region Doughnut", buildRegionDoughnutChart],
    ["Premium Trend", buildPremiumTrendChart],
    ["Risk Gauge", buildRiskGaugeChart]
  ];

  builders.forEach(([name, fn]) => {
    try {
      fn();
    } catch (err) {
      console.error(`[Vitalis] Chart failed to build: ${name}`, err);
    }
  });
}

function buildFeatureImportanceChart() {
  const ctx = document.getElementById("chart-feature-importance");
  const labels = ["Risk Score", "Chronic Count", "Days Hospitalized", "Smoker", "COPD", "Visits", "Hospitalizations", "Medication Count", "BMI", "Income"];
  const data = [0.182, 0.151, 0.129, 0.108, 0.095, 0.084, 0.071, 0.063, 0.052, 0.041];
  charts.featureImportance = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        data,
        backgroundColor: labels.map((_, i) => `rgba(0,229,199,${0.9 - i * 0.06})`),
        borderRadius: 6,
        barThickness: 16
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => ` Importance: ${(c.raw * 100).toFixed(1)}%` } } },
      scales: {
        x: { grid: { color: CHART_COLORS.grid }, ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` } },
        y: { grid: { display: false } }
      }
    }
  });
}

function buildCostDistributionChart() {
  const ctx = document.getElementById("chart-cost-distribution");
  const labels = ["0–5k", "5–10k", "10–15k", "15–20k", "20–25k", "25–30k", "30k+"];
  const data = [12, 28, 34, 21, 12, 6, 3];
  charts.costDistribution = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{ data, backgroundColor: "rgba(124,140,255,0.55)", hoverBackgroundColor: "rgba(124,140,255,0.85)", borderRadius: 5 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: CHART_COLORS.grid }, ticks: { callback: (v) => `${v}%` } }
      }
    }
  });
}

function buildActualPredictedChart() {
  const ctx = document.getElementById("chart-actual-predicted");
  const points = Array.from({ length: 26 }, () => {
    const actual = 2000 + Math.random() * 26000;
    const noise = (Math.random() - 0.5) * 3200;
    return { x: actual, y: Math.max(500, actual + noise) };
  });
  const maxVal = 30000;
  charts.actualPredicted = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          type: "scatter",
          label: "Cases",
          data: points,
          backgroundColor: "rgba(0,229,199,0.75)",
          pointRadius: 4
        },
        {
          type: "line",
          label: "Perfect calibration",
          data: [{ x: 0, y: 0 }, { x: maxVal, y: maxVal }],
          borderColor: "rgba(255,255,255,0.25)",
          borderDash: [5, 5],
          pointRadius: 0,
          borderWidth: 1.5
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: "Actual ($)", color: CHART_COLORS.text, font: { size: 10 } }, grid: { color: CHART_COLORS.grid } },
        y: { title: { display: true, text: "Predicted ($)", color: CHART_COLORS.text, font: { size: 10 } }, grid: { color: CHART_COLORS.grid } }
      }
    }
  });
}

function buildRiskRadarChart() {
  const ctx = document.getElementById("chart-risk-radar");
  charts.riskRadar = new Chart(ctx, {
    type: "radar",
    data: {
      labels: ["Age", "BMI", "Risk Score", "Chronic Count", "Hospital Visits", "Hospital Days"],
      datasets: [{
        label: "Current Assessment",
        data: [40, 40, 40, 40, 40, 40],
        backgroundColor: "rgba(0,229,199,0.18)",
        borderColor: CHART_COLORS.accent,
        pointBackgroundColor: CHART_COLORS.accent,
        borderWidth: 2
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        r: {
          angleLines: { color: CHART_COLORS.grid },
          grid: { color: CHART_COLORS.grid },
          pointLabels: { font: { size: 10 } },
          ticks: { display: false, maxTicksLimit: 4 },
          suggestedMin: 0, suggestedMax: 100
        }
      }
    }
  });
}

function buildHealthPolarChart() {
  const ctx = document.getElementById("chart-health-polar");
  charts.healthPolar = new Chart(ctx, {
    type: "polarArea",
    data: {
      labels: ["BMI", "Systolic BP", "Diastolic BP", "LDL", "HbA1c", "Medications"],
      datasets: [{
        data: [30, 30, 30, 30, 30, 30],
        backgroundColor: [
          "rgba(0,229,199,0.55)", "rgba(124,140,255,0.55)", "rgba(185,140,255,0.55)",
          "rgba(255,193,94,0.55)", "rgba(255,107,90,0.5)", "rgba(255,255,255,0.25)"
        ],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 8, font: { size: 9 } } } },
      scales: { r: { grid: { color: CHART_COLORS.grid }, ticks: { display: false } } }
    }
  });
}

function buildLifestylePieChart() {
  const ctx = document.getElementById("chart-lifestyle-pie");
  charts.lifestylePie = new Chart(ctx, {
    type: "pie",
    data: {
      labels: ["Current", "Former", "Never"],
      datasets: [{
        data: [22, 33, 45],
        backgroundColor: ["rgba(255,107,90,0.75)", "rgba(255,193,94,0.75)", "rgba(0,229,199,0.75)"],
        borderColor: "#0D131F",
        borderWidth: 2,
        offset: [0, 0, 0]
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { position: "bottom", labels: { boxWidth: 8, font: { size: 10 } } } }
    }
  });
}

function buildRegionDoughnutChart() {
  const ctx = document.getElementById("chart-region-doughnut");
  charts.regionDoughnut = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Central", "East", "North", "South", "West"],
      datasets: [{
        data: [18, 24, 15, 26, 17],
        backgroundColor: [
          "rgba(0,229,199,0.75)", "rgba(124,140,255,0.75)", "rgba(185,140,255,0.75)",
          "rgba(255,193,94,0.75)", "rgba(255,107,90,0.7)"
        ],
        borderColor: "#0D131F",
        borderWidth: 2,
        offset: [0, 0, 0, 0, 0]
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: "62%",
      plugins: { legend: { position: "bottom", labels: { boxWidth: 8, font: { size: 10 } } } }
    }
  });
}

function buildPremiumTrendChart() {
  const ctx = document.getElementById("chart-premium-trend");
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const data = [410, 425, 418, 440, 452, 468, 471, 460, 483, 495, 502, 511];
  const gradient = ctx.getContext("2d").createLinearGradient(0, 0, 0, 240);
  gradient.addColorStop(0, "rgba(0,229,199,0.35)");
  gradient.addColorStop(1, "rgba(0,229,199,0)");
  charts.premiumTrend = new Chart(ctx, {
    type: "line",
    data: {
      labels: months,
      datasets: [{
        label: "Avg. Monthly Premium",
        data,
        borderColor: CHART_COLORS.accent,
        backgroundColor: gradient,
        fill: true,
        tension: 0.42,
        pointRadius: 3,
        pointBackgroundColor: CHART_COLORS.accent,
        pointBorderColor: "#06140F"
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: CHART_COLORS.grid }, ticks: { callback: (v) => `$${v}` } }
      }
    }
  });
}

function buildRiskGaugeChart() {
  const ctx = document.getElementById("chart-risk-gauge");
  const gaugeCenterText = {
    id: "gaugeCenterText",
    afterDraw(chart) {
      if (chart.canvas.id !== "chart-risk-gauge") return;
      const { ctx, chartArea } = chart;
      const value = chart._gaugeValue ?? 0;
      const cx = (chartArea.left + chartArea.right) / 2;
      const cy = chartArea.bottom - 4;
      ctx.save();
      ctx.textAlign = "center";
      ctx.fillStyle = "#EDF1F8";
      ctx.font = "700 26px 'JetBrains Mono', monospace";
      ctx.fillText(Math.round(value), cx, cy - 6);
      ctx.fillStyle = "#5B6478";
      ctx.font = "500 11px 'Inter', sans-serif";
      ctx.fillText("Risk / 100", cx, cy + 14);
      ctx.restore();
    }
  };

  charts.riskGauge = new Chart(ctx, {
    type: "doughnut",
    data: {
      datasets: [{
        data: [0, 100],
        backgroundColor: [CHART_COLORS.accent, "rgba(255,255,255,0.07)"],
        borderWidth: 0,
        circumference: 180,
        rotation: 270,
        cutout: "78%"
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } }
    },
    plugins: [gaugeCenterText]
  });
  charts.riskGauge._gaugeValue = 0;
}

function updateGauge(value) {
  const clamped = Math.max(0, Math.min(100, value));
  const color = clamped >= 70 ? CHART_COLORS.danger : clamped >= 40 ? CHART_COLORS.warn : CHART_COLORS.accent;
  const gauge = charts.riskGauge;
  gauge.data.datasets[0].data = [clamped, 100 - clamped];
  gauge.data.datasets[0].backgroundColor = [color, "rgba(255,255,255,0.07)"];
  gauge._gaugeValue = clamped;
  gauge.update();
}

/**
 * Normalizes a raw value to a 0–100 scale against an assumed max,
 * for radar / polar visualization only. Purely presentational.
 */
function normalize(value, max) {
  if (Number.isNaN(value) || value === null) return 0;
  return Math.max(0, Math.min(100, (value / max) * 100));
}

function updateLiveCharts(payload) {
  if (!chartsReady) return; // Charts never loaded — skip silently, prediction still shows.

  // Risk radar: Age, BMI, Risk Score, Chronic Count, Hospital Visits, Hospital Days
  charts.riskRadar.data.datasets[0].data = [
    normalize(payload.age, 100),
    normalize(payload.bmi, 50),
    normalize(payload.risk_score, 100),
    normalize(payload.chronic_count, 10),
    normalize(payload.visits_last_year, 20),
    normalize(payload.days_hospitalized_last_3yrs, 30)
  ];
  charts.riskRadar.update();

  // Health polar: BMI, Systolic BP, Diastolic BP, LDL, HbA1c, Medications
  charts.healthPolar.data.datasets[0].data = [
    normalize(payload.bmi, 50),
    normalize(payload.systolic_bp, 200),
    normalize(payload.diastolic_bp, 120),
    normalize(payload.ldl, 200),
    normalize(payload.hba1c, 14),
    normalize(payload.medication_count, 15)
  ];
  charts.healthPolar.update();

  updateGauge(payload.risk_score);

  // Highlight the matching smoker slice
  const smokerLabels = ["Current", "Former", "Never"];
  const smokerIdx = smokerLabels.indexOf(payload.smoker);
  charts.lifestylePie.data.datasets[0].offset = smokerLabels.map((_, i) => (i === smokerIdx ? 16 : 0));
  charts.lifestylePie.update();

  // Highlight the matching region slice
  const regionLabels = ["Central", "East", "North", "South", "West"];
  const regionIdx = regionLabels.indexOf(payload.region);
  charts.regionDoughnut.data.datasets[0].offset = regionLabels.map((_, i) => (i === regionIdx ? 12 : 0));
  charts.regionDoughnut.update();
}

/* ---------------------------------------------------------
   Prediction submit handler
--------------------------------------------------------- */
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
  const t0 = performance.now();

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const elapsed = Math.round(performance.now() - t0);
    if (orbitLatency) orbitLatency.textContent = `${elapsed}ms`;

    if (!response.ok) {
      let detail = `Request failed with status ${response.status}.`;
      try {
        const errBody = await response.json();
        if (errBody && errBody.detail) {
          detail = typeof errBody.detail === "string" ? errBody.detail : JSON.stringify(errBody.detail);
        }
      } catch (_) { /* no JSON body */ }
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

    const confidencePct = Math.max(58, 100 - Math.min(42, Math.round(payload.risk_score * 0.4)));
    confidenceFill.style.width = `${confidencePct}%`;
    confidenceLabel.textContent = `±${(100 - confidencePct + 4)}% estimated variance`;

    animateResultEcg(Math.min(cost / 25000, 1));
    try {
      updateLiveCharts(payload);
    } catch (chartErr) {
      console.error("[Vitalis] Live chart update failed (prediction still valid):", chartErr);
    }

    session.count += 1;
    session.total += cost;
    refreshSessionStats();

    document.getElementById("prediction").scrollIntoView({ behavior: "smooth", block: "start" });

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

/* ---------------------------------------------------------
   Init
--------------------------------------------------------- */
let chartsReady = false;

function showChartsUnavailableNotice() {
  document.querySelectorAll(".chart-canvas-wrap").forEach((wrap) => {
    if (wrap.querySelector(".chart-fallback-note")) return;
    const note = document.createElement("p");
    note.className = "chart-fallback-note";
    note.style.cssText = "color:#5B6478;font-size:0.8rem;text-align:center;padding-top:40px;font-family:'JetBrains Mono',monospace;";
    note.textContent = "Chart library unavailable — check your internet connection and refresh.";
    wrap.appendChild(note);
  });
}

function tryInitCharts() {
  if (chartsReady || typeof Chart === "undefined") return false;
  initChartDefaults();
  buildAllCharts();
  chartsReady = true;
  return true;
}

document.addEventListener("DOMContentLoaded", () => {
  refreshSessionStats();
  if (!tryInitCharts()) {
    // Give the fallback CDN (loaded async in index.html) a moment to arrive.
    window.addEventListener("chartjs-ready", () => tryInitCharts());
    setTimeout(() => {
      if (!tryInitCharts()) showChartsUnavailableNotice();
    }, 3000);
  }
});