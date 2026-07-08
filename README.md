# 🏥 Medical Insurance Cost Prediction


An end-to-end Machine Learning web application that predicts annual medical insurance costs using a trained **CatBoost Regressor** model, served through a **FastAPI** backend and consumed by an interactive, dark-themed dashboard.

---

## 📌 Project Overview

**Medical Insurance Cost Prediction** is a complete ML-powered web application designed to estimate an individual's annual medical insurance expenses based on personal and lifestyle attributes such as age, BMI, smoking status, and region. 

The project integrates:

- 🤖 A trained **CatBoost Regressor** model
- ⚡ A **FastAPI** backend exposing a REST API
- 🎨 An interactive, responsive frontend dashboard
- 🔄 A structured data preprocessing pipeline
- 📊 Real-time prediction and visualization using Chart.js
- 🛡️ Input validation and robust error handling

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| **Language** | Python |
| **Data Handling** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn, CatBoost Regressor |
| **Backend** | FastAPI, Pydantic |
| **Model Serialization** | Joblib |
| **Frontend** | HTML, CSS, JavaScript |
| **Visualization** | Chart.js |

---

## 🧠 Model Details

| Attribute | Details |
|---|---|
| **Algorithm** | CatBoost Regressor |
| **Task Type** | Regression |
| **Target Variable** | Annual Medical Cost |
| **Serialization** | Joblib (`.pkl`) |

---

## 📈 Model Performance

| Metric | Value | What It Means |
|---|---|---|
| **MAE** (Mean Absolute Error) | `1767.70` | On average, the model's predicted insurance cost differs from the actual cost by about **$1,767.70**. Lower is better — this gives an intuitive sense of typical prediction error in real dollar terms. |
| **RMSE** (Root Mean Squared Error) | `2843.58` | Similar to MAE, but penalizes larger errors more heavily. An RMSE of **$2,843.58** indicates the presence of some larger deviations between predicted and actual costs. |
| **R² Score** | `0.1783` | The model explains approximately **17.83%** of the variance in medical costs. This suggests there is room for improvement — likely through additional features, hyperparameter tuning, or more advanced preprocessing. |

> 💡 **Note:** These metrics reflect an initial baseline model. See the [Future Improvements](#-future-improvements) section for planned enhancements to boost predictive performance.

---

## ✨ Features

- 🖥️ Professional, dark-themed dashboard UI
- ⚡ FastAPI-powered REST API
- 🤖 CatBoost Machine Learning model integration
- 🔮 Real-time cost predictions
- ✅ Strict input validation via Pydantic
- 📱 Fully responsive design
- 📊 Interactive charts and data visualization
- 🎨 Clean, modern, professional design language
- 🛡️ Comprehensive error handling
- 💾 Model persistence using Joblib

---

## 📁 Project Structure

```
medical-insurance-cost-prediction/
│
├── app/
│   ├── main.py                  # FastAPI application entry point
│   ├── model/
│   │   ├── catboost_model.pkl   # Serialized trained model
│   │   └── preprocessor.pkl     # Serialized preprocessing pipeline
│   ├── schemas/
│   │   └── request_schema.py    # Pydantic request/response models
│   └── utils/
│       └── preprocessing.py     # Data preprocessing utilities
│
├── frontend/
│   ├── index.html                # Dashboard UI
│   ├── style.css                 # Styling
│   └── script.js                 # Frontend logic & API calls
│
├── data/
│   └── insurance.csv             # Training dataset
│
├── notebooks/
│   └── model_training.ipynb      # EDA & model training notebook
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── LICENSE                        # License file
```

---

## ⚙️ Installation

Follow the steps below to set up the project locally.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/medical-insurance-cost-prediction.git
cd medical-insurance-cost-prediction
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the FastAPI Backend

```bash
uvicorn main:app --reload
```

The API will now be running at:
```
http://127.0.0.1:8000
```

Interactive API docs (Swagger UI) are available at:
```
http://127.0.0.1:8000/docs
```

### 5️⃣ Launch the Frontend

Navigate to the `frontend/` directory and open `index.html` directly in your browser, or serve it using a lightweight local server:

```bash
cd frontend
python -m http.server 5500
```

Then visit:
```
http://127.0.0.1:5500
```

Ensure the frontend's API base URL points to your running FastAPI backend (`http://127.0.0.1:8000`).

---

## 📡 API Documentation

### `GET /`

**Description:** Health check endpoint to verify the API is running.

**Response:**
```json
{
  "message": "Medical Insurance Cost Prediction API is running"
}
```

---

### `POST /predict`

**Description:** Accepts user input features and returns a predicted annual medical insurance cost.

**Request Body:**
```json
{
  "age": 34,
  "sex": "male",
  "bmi": 27.5,
  "children": 2,
  "smoker": "no",
  "region": "southeast"
}
```

| Field | Type | Description |
|---|---|---|
| `age` | integer | Age of the individual |
| `sex` | string | Gender (`male` / `female`) |
| `bmi` | float | Body Mass Index |
| `children` | integer | Number of dependents |
| `smoker` | string | Smoking status (`yes` / `no`) |
| `region` | string | Residential region |

**Response:**
```json
{
  "predicted_cost": 8523.47
}
```

| Field | Type | Description |
|---|---|---|
| `predicted_cost` | float | Predicted annual medical insurance cost in USD |

---

## 🔄 Workflow

1. **Dataset**
   Raw insurance data (age, sex, BMI, children, smoker status, region, charges) is collected and loaded for analysis.

2. **Preprocessing**
   Categorical variables are encoded, missing values are handled, and features are prepared into a format suitable for model training.

3. **Model Training**
   A CatBoost Regressor is trained on the preprocessed dataset to learn the relationship between input features and medical costs.

4. **Model Saving**
   The trained model and preprocessing pipeline are serialized using **Joblib** for efficient reuse during inference.

5. **FastAPI Integration**
   The serialized model is loaded into a FastAPI application, which exposes a `/predict` endpoint for real-time inference.

6. **Frontend**
   A responsive HTML/CSS/JavaScript dashboard collects user input, sends requests to the API, and visualizes predictions using Chart.js.

7. **Prediction Flow**
   ```
   User Input → Frontend Validation → API Request → 
   Preprocessing → Model Inference → Response → 
   Dashboard Visualization
   ```

---

## 🚀 Future Improvements

- 🐳 **Docker** containerization for consistent deployment environments
- ☁️ **Cloud Hosting** (AWS / Azure / GCP / Render)
- 🔐 **Authentication** for secure API access
- 🗄️ **Database Integration** to store prediction history
- 📉 **Model Monitoring** to track performance drift over time
- 🔁 **CI/CD Pipelines** for automated testing and deployment
- 📊 Improved feature engineering to boost R² score

---

## 👤 Author

**Ayush Palse**
*Machine Learning Enthusiast*

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it with attribution.

---

<div align="center">

⭐ If you found this project useful, consider giving it a star on GitHub!

</div>