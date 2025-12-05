# Advanced Football Predictor -- Poisson xG & Machine Learning

This repository contains an advanced football prediction and analytics
dashboard built with Python and Streamlit. The application combines
Poisson expected goals (xG) modelling with multiple machine learning
algorithms to generate structured, data-driven insights for football
matches.

Live and historical match data are pulled from external APIs
(football-data.org and API-FOOTBALL) and transformed into rolling
performance features such as goals for, goals against and goal
difference per game over a configurable form window. These engineered
features feed into a range of predictive models including Poisson xG,
Logistic Regression, Random Forests, XGBoost, and Support Vector
Machines / Regressors.

The multi-page Streamlit interface includes match overviews with xG and
win/draw/loss probabilities, a Bet Insights module projecting goal
lines, multigoals and corners (for educational analysis only), confusion
matrices and performance metrics, feature importance and explainable AI
summaries, Monte-Carlo match simulations, xG confidence intervals,
time-series form trends, model comparison dashboards and downloadable
datasets. A dedicated player analysis section is also powered by
API-FOOTBALL.

Technically, this project demonstrates real-world skills in Python,
pandas, NumPy, scikit-learn, XGBoost, Plotly, Streamlit, API
integration, caching, feature engineering, model evaluation and
interactive dashboard design. It is intended as a professional portfolio
project showcasing applied data science and sports analytics
capabilities.
