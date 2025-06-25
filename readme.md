# Housing Prices Prediction

[Kaggle competition to predict housing prices based on 79 features.](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)  Submission achieved **top 2% performance** out of ~24,000 submissions.

## Project Overview

This project implements an end-to-end machine learning pipeline for house price prediction, featuring sophisticated data preprocessing, domain-knowledge-driven feature engineering, and comprehensive model evaluation across 8 different algorithms.

## Key Results

- Top 2% Performance - Achieved excellent predictive accuracy
- Comprehensive Model Comparison - Evaluated 8 different ML algorithms
- Advanced Feature Engineering - Domain-based imputation and Box-Cox transformations
- Robust Validation - 5-fold cross-validation for reliable performance estimates

## Tech Stack

- **Data Science & ML**: Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM  
- **Visualization**: Matplotlib
- **Statistical Analysis**: SciPy  
- **Development**: Jupyter Notebooks, Git

## Project Structure

```
housing-prices-prediction/
├── .vscode                                         # IDE preferences and formatting rules
├── raw-data/                                       
│   ├── train.csv                                   # Training dataset
│   ├── test.csv                                    # Test dataset
│   └── data_description.txt                        # Feature descriptions
├── processed/
│   ├── processed_train.csv                         # Cleaned training data
│   └── processed_test.csv                          # Cleaned test data
├── 01_house_prices_eda_and_preprocessing.ipynb     # Exploratory Data Analysis
├── 02_house_prices_model_selection.ipynb           # Model Development & Evaluation
├── submission.csv                                  # Final predictions
└── requirements.txt                                # Dependencies
```

## Installation & Setup

Kaggle's data is already in the `raw-data` directory.  [You can also download it from the competition's data tab.](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

1. **Clone the repository**
   ```bash
   git clone https://github.com/wh33les/housing-prices-prediction.git
   cd housing-prices-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the analysis in Jupyter**

   - Step 1: Data preprocessing and EDA
   
   01_house_prices_eda_and_preprocessing.ipynb
   
   - Step 2: Model training and evaluation
   
   02_house_prices_model_selection.ipynb
   ```

## Methodology

### 1. **Data Preprocessing & Feature Engineering**

- Domain-Based Imputation: Leveraged real estate knowledge to handle missing values
- Ordinal Encoding: Converted quality ratings to numerical scales
- Logical Corrections: Applied business rules for feature consistency
- Skewness Correction: Box-Cox transformations for 15+ highly skewed features
- Feature Creation: Added `TotalSF`, `TotalBathrooms`, `HouseAge`, `RemodAge`

### 2. **Model Development**

- 8 Algorithm Comparison: Linear Regression, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost, LightGBM
- Robust Scaling: Applied to linear models for optimal performance
- Target Transformation: Log transformation to normalize price distribution
- Cross-Validation: 5-fold CV for unbiased performance estimation

### 3. **Model Selection & Evaluation**

- Comprehensive Metrics: RMSE, MAE, R² across all models
- Feature Importance Analysis: Identified key price predictors
- Residual Analysis: Validated model assumptions
- Final Ensemble: Selected best-performing algorithm based on CV scores

## Performance Metrics

1. Lasso CV
- Cross-val RMSE: 0.1261 (+/- 0.0151)
- Top 3 important features: TotalSF, OverallQual, Neighborhood_Crawfor

2. ElasticNet CV
- Cross-val RMSE: 0.1263 (+/- 0.0150)
- Top 3 important features: TotalSF, OverallQual, Neighborhood_StoneBr

3. Gradient Boosting
- Cross-val RMSE: 0.1272 (+/- 0.0081)
- Top 3 important features: TotalSF, OverallQual, TotalBathrooms

## Visualizations

The project includes comprehensive visualizations:

- Feature Distributions
- Correlation Analysis
- Model Performance
- Residual Analysis

## Future Enhancements

- Advanced Ensembles: Stacking and blending techniques  
- Feature Selection* Recursive feature elimination
- Deep Learning: Neural network architectures
- Real-time Predictions: API deployment with Flask/FastAPI

## Contact & Collaboration

**Interested in data science collaboration or have questions about this project?**

- **Email**: [ashleykwwarren@gmail.com]
- **LinkedIn**: [Ashley K. W. Warren](https://www.linkedin.com/in/ashleykwwarren/)

---

## Keywords 

**Machine Learning**: `Regression` `Ensemble Methods` `Cross-Validation` `Model Selection` `Hyperparameter Tuning` `Feature Engineering` `Predictive Modeling`

**Algorithms**: `Linear Regression` `Ridge Regression` `Lasso Regression` `ElasticNet` `Random Forest` `Gradient Boosting` `XGBoost` `LightGBM`

**Data Science**: `Exploratory Data Analysis` `Feature Scaling` `Data Preprocessing` `Missing Value Imputation` `Outlier Detection` `Statistical Analysis`

**Statistical Methods**: `Box-Cox Transformation` `Log Transformation` `Skewness Correction` `Correlation Analysis` `Distribution Analysis`

**Python Libraries**: `pandas` `numpy` `scikit-learn` `xgboost` `lightgbm` `matplotlib` `seaborn` `scipy`

**Evaluation Metrics**: `RMSE` `MAE` `R-squared` `Cross-Validation Score` `Residual Analysis`

**Domain Knowledge**: `Real Estate` `Housing Market` `Property Valuation` `Domain-Based Feature Engineering`

**Software Engineering**: `Git Version Control` `Code Organization` `Reproducible Research` `Documentation` `Data Pipeline`