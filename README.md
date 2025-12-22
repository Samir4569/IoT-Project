# IoT Energy Consumption Prediction

## Overview

This project analyzes and predicts household appliance energy consumption using machine learning and deep learning techniques. The dataset (`KAG_energydata_complete.csv`) contains time-series measurements of energy usage, indoor environmental factors, and timestamps. The goal is to understand consumption patterns and build predictive models to estimate appliance energy usage accurately.

## Features & Preprocessing

* **Date-Time Features:** Extracted hour, day of the week, and month to capture temporal patterns.
* **Environmental Data:** Temperature, humidity, and other indoor sensor readings.
* **Target Variable:** `Appliances` energy consumption in Wh.
* **Feature Scaling:** Standardized features for neural networks and linear regression.
* **Log Transformation:** Applied to the target variable to reduce skewness for neural network training.

## Modeling Approaches

The project implements and evaluates multiple models:

1. **Linear Regression** – Baseline model to capture linear relationships.
2. **Random Forest** – Handles non-linear patterns and skewed peaks efficiently.
3. **XGBoost** – Gradient boosting model with potential for hyperparameter tuning.
4. **MLP Neural Network** – Deep learning approach using log-transformed targets.

### Model Evaluation Metrics

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**
* **R² Score**

| Model              | MAE   | RMSE    | R²   | Comment                                                            |
| ------------------ | ----- | ------- | ---- | ------------------------------------------------------------------ |
| Linear Regression  | 52.55 | 8297.32 | 0.17 | Baseline, poor fit for non-linear patterns.                        |
| Random Forest      | 31.07 | 4412.01 | 0.56 | Best-performing, captures non-linear patterns and peaks well.      |
| XGBoost            | 37.34 | 5454.08 | 0.45 | Slightly worse than Random Forest, tunable for better performance. |
| MLP Neural Network | 34.94 | 6433.38 | 0.36 | Better than Linear Regression but struggles with extreme peaks.    |

## Visualization

* Distribution plots of energy consumption and log-transformed targets.
* Hourly average energy consumption trends.
* Actual vs predicted values for Random Forest.
* Training and validation loss curves for the neural network.

