# AI/ML Capstone Project

This repository contains three independent modules developed as part of the AI/ML capstone project:

1. **Data Pipeline**
2. **Analytics and Machine Learning**
3. **Support Assistant**

Each module contains its own implementation and supporting files.

---

## Repository Structure

```text
project-root/
│
├── README.md
│
├── data-pipeline/
│   ├── ...
│   └── README.md
│
├── analytics/
│   ├── ...
│   └── README.md
│
└── support-assistant/
    ├── ...
    └── README.md
```

---

# 1. Data Pipeline

The Data Pipeline module demonstrates web scraping, data cleaning, currency conversion, SQLite database creation, SQL querying, and pandas-based analysis.

The implementation uses:

* Python
* Requests
* BeautifulSoup
* pandas
* SQLite

The source website used by the implementation is **Books to Scrape**.

The pipeline extracts book information including:

* Book title
* Price in GBP
* Star rating
* Availability
* Category

The price is converted from GBP to INR using the fixed conversion rate specified in the project:

```text
1 GBP = 105.50 INR
```

The cleaned data is stored in a SQLite database using separate category and book tables.

The module also demonstrates SQL queries involving filtering, grouping, ordering, distinct values, limits, joins, and range conditions.

---

# 2. Analytics

The Analytics module uses the Titanic dataset for exploratory data analysis and machine learning.

The implementation includes:

* Dataset loading
* Dataset profiling
* Missing-value analysis
* Data cleaning
* Univariate analysis
* Bivariate analysis
* Correlation analysis
* Data visualization
* Feature preparation
* Train/test splitting
* Classification
* Class-imbalance comparison
* Random Forest hyperparameter tuning
* Regression analysis
* Model saving

The classification models implemented are:

* Logistic Regression
* Decision Tree
* Random Forest

The classification models are evaluated using metrics including:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC/AUC
* Confusion Matrix

The implementation also compares approaches for handling class imbalance, including:

* Baseline classification
* `class_weight="balanced"`
* SMOTE

A Random Forest model is tuned using `GridSearchCV`.

The regression portion uses linear regression to predict fare and reports regression metrics including:

* MAE
* RMSE
* R²
* Adjusted R²

The final trained pipeline is saved using `joblib`.

---

# 3. Support Assistant

The Support Assistant module implements a retrieval-based support assistant using Zepto policy documents.

The implementation uses:

* Python
* Sentence Transformers
* `all-MiniLM-L6-v2`
* ChromaDB
* LangGraph
* Pydantic
* FastAPI

The assistant embeds policy text and stores the embeddings in ChromaDB.

When a user submits a question, the application first determines whether the question is related to a policy.

Policy-related queries are processed using retrieval from the embedded document collection.

General questions are handled separately.

The LangGraph workflow contains nodes for:

* Intent classification
* Retrieval and answer generation
* Direct answer generation

The API exposes a FastAPI endpoint:

```text
POST /ask
```

The response is structured using a Pydantic model.

The implementation also supports a `MOCK_LLM` mode for deterministic local execution.

---

# Running the Project

Each module is implemented independently.

## Data Pipeline

Navigate to:

```bash
cd data-pipeline
```

Open the notebook or execute the available Python implementation.

## Analytics

Navigate to:

```bash
cd analytics
```

Execute the notebooks in the order provided by the module.

## Support Assistant

Navigate to:

```bash
cd support-assistant
```

The FastAPI application can be started using the command implemented in the support assistant Python file.

---

# Technologies Used

| Module            | Main Technologies                                                          |
| ----------------- | -------------------------------------------------------------------------- |
| Data Pipeline     | Python, Requests, BeautifulSoup, pandas, SQLite                            |
| Analytics         | Python, pandas, NumPy, Seaborn, Matplotlib, scikit-learn, imbalanced-learn |
| Support Assistant | Sentence Transformers, ChromaDB, LangGraph, Pydantic, FastAPI              |

---

# Project Notes

Each module is maintained separately so that the individual components can be executed and evaluated independently.

The notebooks contain the detailed implementation, analysis, outputs, and visualizations for their respective modules.
