# House Prices - Advanced Regression Techniques

Machine learning pipeline for predicting residential property sale prices using feature engineering and ensemble methods.

[Kaggle Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) | Python 3.13 | `scikit-learn` | `XGBoost`

## Overview

This project implements a complete ML workflow for the Kaggle House Prices competition. The goal is to predict sale prices of homes in Ames, Iowa using 79 explanatory variables. 

__Final Result:__ Placed 458 out of 24,509 submissions (top 1.9%), with RMSE of 0.12203.

## Implementation

### (1) Data Exploration and Preprocessing

__File:__ `01_house_prices_eda_and_preprocessing.ipynb`

Started with exploratory data analysis of the 79 features. Key findings:
- Target variable (`SalePrice`) is right-skewed, applied log transformation.
- Missing values in 4 key features: `LotFrontage`, `MasVnrArea`, `BsmtQual`, `GarageYrBlt`.
- 25+ numerical features were skewed, applied Box-Cox transformation.
- Categorical features required ordinal encoding for quality ratings.

__Preprocessing pipeline:__
- Missing value imputation using domain knowledge (no garage = 0 cars, no basement = 0 area).
- Ordinal encoding for quality features (Poor=1, Excellent=5).
- Box-Cox transformation for skewed numerical features.
- One-hot encoding for categorical features.
- Feature engineering: `TotalSF`, `TotalBathrooms`, `HouseAge`, `RemodAge`.
- `RobustScaler` for feature scaling.

### (2) Model Selection and Training

__File:__ `02_house_prices_model_selection.ipynb`

Tested 8 different algorithms using 5-fold cross-validation:

__Linear Models:__
- Linear Regression (baseline)
- Ridge Regression (&alpha;=1.0)
- Lasso Regression (&alpha;=0.001)
- Elastic Net with CV hyperparameter tuning

__Tree-Based Models:__
- Random Forest (`n_estimators=100`)
- Gradient Boosting (`n_estimators=100`)
- XGBoost (`n_estimators=100`)
- LightGBM (`n_estimators=100`)

__Model Comparison with CV RMSE:__

- Elastic Net: 0.1256
- Ridge Regression: 0.1289
- Random Forest: 0.1312
- Gradient Boosting: 0.1356
- XGBoost: 0.1378
- LightGBM: 0.1394
- Lasso Regression: 0.1402
- Linear Regression: 0.1445

__Results:__ Elastic Net performed best with cross-validation RMSE of 0.1256. Key features were `OverallQual`, `GrLivArea`, `Functional`, `Neighborhood_StoneBr`, `TotalSF`.

### (3) Model Implementation

The final model uses `ElasticNet` with the following configuration:

- Used scaled features (204 features after one-hot encoding)
- Log-transformed target variable
- Cross-validated hyperparameters: `alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100]`, `l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9]`
- 5-fold cross-validation for performance estimation
- Coefficient analysis for feature importance

## Implementation Notes

The preprocessing pipeline handles the main data quality issues:
- Domain-specific missing value imputation prevents information loss.
- Box-Cox transformation addresses skewness in numerical features.
- Proper train/test alignment for one-hot encoded features.
- `RobustScaler` handles outliers better than `StandardScaler`.

Model selection revealed that regularized linear models outperform tree-based methods for this dataset. The `ElasticNet` regularization effectively handles the high-dimensional feature space after one-hot encoding. Feature scaling with `RobustScaler` was crucial for linear model performance.

## Future Work

- __Next:__ Experiment with neural networks and `PyTorch`
- Polynomial features and feature interactions
- Feature selection using recursive feature elimination
- Ensemble methods (stacking multiple algorithms)
- Different regularization approaches (Ridge with different alphas)