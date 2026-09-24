# Analytics and Machine Learning

## Overview

This module performs exploratory data analysis and machine learning using the Titanic dataset.

The implementation is divided into two main notebooks:

```text
01_eda.ipynb
02_modeling.ipynb
```

The EDA notebook performs data profiling, cleaning, exploratory analysis, visualization, and correlation analysis.

The modeling notebook performs classification, class-imbalance analysis, Random Forest tuning, and regression.

---

# Dataset

The implementation uses the Titanic dataset available through the Seaborn dataset collection.

The dataset contains passenger information such as:

* Survival
* Passenger class
* Sex
* Age
* Number of siblings/spouses aboard
* Number of parents/children aboard
* Fare
* Embarked information

---

# Exploratory Data Analysis

The EDA implementation includes:

* Dataset information
* Dataset dimensions
* Descriptive statistics
* Missing-value analysis
* Data cleaning
* Univariate analysis
* Bivariate analysis
* Multivariate visualization
* Correlation analysis

---

# Missing Values

The notebook calculates missing-value information and applies cleaning/imputation operations to the dataset.

The cleaned dataset is then used for subsequent exploratory analysis.

---

# Univariate Analysis

The notebook analyzes numerical variables including:

* Age
* Fare

Histograms and box plots are used to understand their distributions.

The notebook also uses the Interquartile Range (IQR) method to identify potential outliers.

---

# Bivariate Analysis

Survival is analyzed against:

* Sex
* Passenger class
* Sex and passenger class together

The notebook uses filtering and grouping operations to calculate and visualize survival-related patterns.

---

# Correlation Analysis

A correlation matrix is created for the main numerical variables.

The implementation examines relationships involving:

```text
survived
pclass
age
sibsp
parch
fare
```

A heatmap is used to visualize the correlation matrix.

The notebook also identifies strong correlations using the absolute value of correlation coefficients.

---

# Data Visualization

The EDA notebook contains multiple visualizations to explore relationships in the Titanic dataset.

These include distribution plots, box plots, survival comparisons, and correlation visualizations.

The visualizations are used to identify patterns involving passenger characteristics and survival.

---

# Machine Learning

The modeling notebook uses a train/test split with stratification for the classification task.

The target variable is:

```text
survived
```

The classification models implemented are:

1. Logistic Regression
2. Decision Tree
3. Random Forest

---

# Preprocessing

The modeling implementation uses scikit-learn preprocessing components.

The preprocessing includes:

* Numerical feature scaling using `StandardScaler`
* Categorical feature encoding using `OneHotEncoder`

A `ColumnTransformer` is used to apply preprocessing to different feature types.

---

# Classification Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC/AUC
* Confusion Matrix

The ROC curves are also visualized.

---

# Class Imbalance

The notebook examines the distribution of the target classes.

Different approaches are compared:

### Baseline

The classifier is trained without explicit class balancing.

### Balanced Class Weights

The classifier uses:

```python
class_weight="balanced"
```

### SMOTE

SMOTE is used to oversample the minority class during model training.

The resulting models are compared using classification metrics.

---

# Random Forest Hyperparameter Tuning

The Random Forest model is tuned using:

```text
GridSearchCV
```

The search considers parameters including:

* Number of estimators
* Maximum tree depth
* Maximum features

The Random Forest implementation also uses:

```python
oob_score=True
```

to obtain an out-of-bag score.

---

# Regression

The modeling notebook also includes a regression task.

A linear regression model is used to predict:

```text
fare
```

using other available features.

The regression model is evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R²
* Adjusted R²

A residual plot is generated to examine the regression errors.

---

# Saved Model

The implementation saves the trained machine-learning pipeline using Joblib.

The saved artifact is:

```text
best_titanic_pipeline.pkl
```

The saved object contains the preprocessing and trained model pipeline.

---

# Files

```text
analytics/
│
├── 01_eda.ipynb
├── 02_modeling.ipynb
├── titanic_raw.csv
├── titanic_cleaned.csv
└── best_titanic_pipeline.pkl
```

The notebooks contain the detailed code, analysis, visualizations, model evaluation, and outputs.

---

# Execution

Run the notebooks in the following order:

```text
01_eda.ipynb
      ↓
02_modeling.ipynb
```

The second notebook uses the cleaned Titanic data produced during the analysis workflow.
