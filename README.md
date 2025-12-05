# ⚽ Advanced Football Prediction Dashboard

An advanced, interactive football analytics and prediction dashboard
built with **Python, Streamlit, Poisson xG modelling, and Machine
Learning**. This project combines statistical modelling with real
football data from live APIs to deliver match predictions, betting
insights, performance analysis, and explainable AI metrics.

This application allows users to select any two professional teams and
instantly generate: - Expected Goals (xG) - Win / Draw / Loss
probabilities - Over/Under goal markets - Corner kick predictions -
Feature importance - Model performance evaluation - Monte-Carlo match
simulations

It is designed for **data science portfolios, sports analytics research,
and machine learning experimentation**.

------------------------------------------------------------------------

## 🚀 Key Features

-   ✅ Live football data via **Football-Data.org API**
-   ✅ Player & team stats via **API-Football (API-Sports)**
-   ✅ **Poisson xG probabilistic model**
-   ✅ **Machine Learning Models**:
    -   Logistic Regression\
    -   Random Forest (Classifier & Regressor)\
    -   XGBoost\
    -   Support Vector Machine (SVM / SVR)
-   ✅ **Bet Confidence Insights** (Goals, Corners, Double Chance, BTTS)
-   ✅ **Monte-Carlo Match Simulator**
-   ✅ **Model Comparison Dashboard**
-   ✅ **Feature Importance & Explainable AI**
-   ✅ **Reliability (Calibration) Diagrams**
-   ✅ **CSV Data Export**

------------------------------------------------------------------------

## 🛠️ Tech Stack

-   **Python 3.10+**
-   **Streamlit**
-   **Scikit-learn**
-   **XGBoost**
-   **NumPy, Pandas**
-   **Plotly**
-   **Football-Data.org API**
-   **API-Football (API-Sports)**

------------------------------------------------------------------------

## 📦 Installation

``` bash
git clone https://github.com/Mark-chidozie/advanced-football-prediction-dashboard.git
cd advanced-football-prediction-dashboard
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🔐 API Setup (IMPORTANT)

Create a file called `secrets.toml` and add:

``` toml
FOOTBALL_DATA_TOKEN = "YOUR_FOOTBALL_DATA_API_KEY"
API_FOOTBALL_KEY = "YOUR_API_FOOTBALL_KEY"
```

⚠️ **Never upload real API keys to GitHub.**

------------------------------------------------------------------------

## ▶️ Run the App

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 📊 Models Implemented

-   Poisson Expected Goals Model\
-   Logistic Regression (Multiclass)\
-   Random Forest (Classifier & Regressor)\
-   XGBoost (Classifier & Regressor)\
-   Support Vector Machines (SVC, SVR)

------------------------------------------------------------------------

## ⚠️ Disclaimer

This project is for **educational and analytical purposes only**. All
probabilities are mathematically derived and **do not constitute
gambling advice**.

------------------------------------------------------------------------

## 👤 Author

**Mark Chidozie**\
MSc Data Science \| Sports Analytics \| Machine Learning\
GitHub: https://github.com/Mark-chidozie
