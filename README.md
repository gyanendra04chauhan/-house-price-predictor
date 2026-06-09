# 🏠 House Price Predictor

<div align="center">

A machine learning web application that predicts **house sale prices** using **Linear Regression**, built with Python, Scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)


</div>

---

## 🔗 Live Demo
(https://hg6fhanhwm3bqered95rht.streamlit.app/)

<div align="center">

👉 **[Try it live on Streamlit Cloud](YOUR_APP_URL_HERE)**

</div>

---

## 📸 App Preview

> Enter property details like lot area, year built, building type, and more — get an instant price estimate with confidence range!

---

## 📊 Model Performance

| Metric | Value | Description |
|--------|-------|-------------|
| 📈 R² Score | **0.37** | Variance explained by model |
| 💰 MAE | **$30,829** | Average prediction error |
| 📉 RMSE | **$41,138** | Root mean squared error |

> Dataset contains **2,919 records** with **1,459 missing SalePrice values** (test set) — model trained on available labeled data only.

---

## ✨ Features

- 🔮 **Instant Price Prediction** — real-time estimate based on 11 property features
- 📊 **Dataset Explorer** — browse raw Ames Housing data inside the app
- 📈 **Price Distribution Chart** — visual overview of sale price spread
- 🎯 **Confidence Range** — prediction ± MAE shown for every result
- 🧾 **Input Summary Table** — see all your entered values at a glance
- 📱 **Clean Responsive UI** — sidebar metrics, two-column layout, gradient result card

---

## 🧠 ML Pipeline

```
Raw CSV Data
     │
     ▼
Null Handling (fillna mean + dropna)
     │
     ▼
One-Hot Encoding (get_dummies)
     │
     ▼
Train / Test Split (80% / 20%)
     │
     ▼
StandardScaler (feature normalization)
     │
     ▼
Linear Regression Model
     │
     ▼
Predicted Sale Price 💰
```

---

## 📋 Features Used for Prediction

| Feature | Type | Description |
|---------|------|-------------|
| `MSSubClass` | Numerical | Building class |
| `MSZoning` | Categorical | Zoning classification |
| `LotArea` | Numerical | Lot size in sq ft |
| `LotConfig` | Categorical | Lot configuration |
| `BldgType` | Categorical | Type of dwelling |
| `OverallCond` | Numerical | Overall condition rating (1–9) |
| `YearBuilt` | Numerical | Original construction year |
| `YearRemodAdd` | Numerical | Remodel year |
| `Exterior1st` | Categorical | Exterior covering material |
| `BsmtFinSF2` | Numerical | Type 2 finished basement area |
| `TotalBsmtSF` | Numerical | Total basement area in sq ft |

---

## 🖥️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/gyanendra04chauhan/-house-price-predictor.git
cd -house-price-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser. 🚀

---

## 📁 Project Structure

```
-house-price-predictor/
│
├── app.py                     # 🎯 Main Streamlit app (UI + ML pipeline)
├── HousePricePrediction.csv   # 📂 Ames Housing dataset (2,919 rows)
├── requirements.txt           # 📦 Python dependencies
├── .gitignore                 # 🚫 Files to ignore
└── README.md                  # 📖 Project documentation
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.9+ | Core programming language |
| Streamlit | Web UI framework |
| Scikit-learn | ML model & preprocessing |
| Pandas | Data manipulation |
| NumPy | Numerical operations |

---

## ☁️ Deploy to Streamlit Cloud

1. Fork / clone this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select this repo → set `app.py` as main file
4. Click **Deploy** — live in ~2 minutes! ✅

---

## 📚 Dataset

- **Name:** Ames Housing Dataset
- **Source:** House Price Prediction (Kaggle-style)
- **Records:** 2,919 total (1,460 train + 1,459 test)
- **Target Variable:** `SalePrice` (USD)

---

## 👨‍💻 Author

**Gyanendra Chauhan**
B.Tech CSE (AI & ML) — Axis Institute of Technology & Management, Kanpur (AKTU, 2023–2027)

[![GitHub](https://img.shields.io/badge/GitHub-gyanendra04chauhan-black?logo=github)](https://github.com/gyanendra04chauhan)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use and modify!

---

<div align="center">
⭐ If you found this project helpful, consider giving it a star!
</div>
