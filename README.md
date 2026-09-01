# Credit Risk - Loan Default Prediction - Machine Learning Model

## Project Overview

This project develops an end-to-end **Machine Learning solution for Credit Risk Assessment and Loan Default Prediction** using the German Credit dataset.

The objective is to predict whether a customer represents **Good or Bad credit risk** based on borrower and credit-related characteristics.

The project combines **credit risk concepts with machine learning techniques**, covering the complete modelling lifecycle from data understanding and preprocessing to model development, evaluation, optimization, explainability, and final risk interpretation.

---

## Business Objective

The objective is to develop a predictive credit risk model that can help identify customers with a higher likelihood of being classified as **Bad Credit Risk**.

From a credit risk perspective, particular attention is given to the model's ability to identify Bad customers, since failing to detect a high-risk borrower can result in increased credit losses.

The project demonstrates how machine learning can support:

* Data-driven credit risk assessment
* Risk segmentation
* Credit decision-making
* Identification of higher-risk borrowers
* More consistent evaluation of borrower creditworthiness

---

## Dataset

The project uses the **German Credit dataset**, containing **1,000 observations and 20 input features** describing different aspects of borrowers' financial and credit profiles.

The target variable is:

* **0 = Bad Credit Risk**
* **1 = Good Credit Risk**

The dataset contains a combination of:

* Numerical features
* Ordinal features
* Categorical features

### Target Distribution

| Credit Risk | Customers | Percentage |
| ----------- | --------: | ---------: |
| Good (1)    |       700 |        70% |
| Bad (0)     |       300 |        30% |

---

## Data Understanding & Quality Checks

The dataset was examined to understand its structure and data quality.

The analysis included:

* Dataset dimensions and structure
* Sample records
* Data types
* Missing value analysis
* Duplicate record analysis
* Target variable distribution
* Number of unique values per feature

### Data Quality Results

* **Rows:** 1,000
* **Input Features:** 20
* **Missing Values:** 0
* **Duplicate Records:** 0

---

## Feature Classification

The input variables were classified according to their characteristics and business meaning to ensure that appropriate preprocessing techniques were applied.

### Numerical Features

* Loan Duration
* Credit Amount
* Age

### Ordinal Features

* Checking Account Status
* Credit History
* Savings Account
* Employment Duration
* Installment Rate
* Residence Duration
* Number of Credits
* Job

### Categorical Features

* Loan Purpose
* Personal Status & Sex
* Other Debtors
* Property
* Other Installment Plans
* Housing
* Telephone
* Foreign Worker

---

## Machine Learning Workflow

The project follows an end-to-end credit modelling workflow:

1. Data Loading
2. Data Understanding
3. Data Quality Assessment
4. Target Variable Analysis
5. Feature Exploration
6. Feature and Target Definition
7. Feature Classification
8. Train/Test Split
9. Data Preprocessing
10. Model Development
11. Model Evaluation
12. Model Comparison
13. Classification Threshold Optimization
14. Model Selection
15. Feature Explainability
16. Hyperparameter Tuning
17. Final Model Evaluation
18. Risk Threshold Optimization
19. Credit Risk Interpretation

---

## Train/Test Split

The dataset was divided into:

* **80% Training Set:** 800 observations
* **20% Test Set:** 200 observations

A **stratified split** was used to preserve the Good/Bad credit risk distribution across the training and test datasets.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

---

## Data Preprocessing

Different preprocessing techniques were applied according to feature type.

| Feature Type | Preprocessing Technique |
| ------------ | ----------------------- |
| Numerical    | StandardScaler          |
| Ordinal      | OrdinalEncoder          |
| Categorical  | OneHotEncoder           |

A **ColumnTransformer** was used to combine the preprocessing steps into a unified transformation pipeline.

This approach ensures that each feature type is processed appropriately before being used by the machine learning models.

---

## Machine Learning Models

Three classification models were developed and evaluated:

### 1. Logistic Regression

Logistic Regression was used as the baseline model due to its interpretability and suitability for binary credit risk classification.

### 2. Decision Tree

A Decision Tree was developed as a non-linear alternative to evaluate whether tree-based decision rules could improve predictive performance.

### 3. Random Forest

A Random Forest model was developed as an ensemble-based alternative combining multiple decision trees.

---

## Model Evaluation

Model performance was evaluated using several classification metrics, with particular attention to the **Bad Credit Risk class**.

The evaluation included:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC

### Why Bad Recall Matters

In credit risk modelling, identifying Bad customers is particularly important.

Therefore, **Bad Recall** was specifically monitored to assess how effectively the model identifies customers who belong to the Bad Credit Risk class.

---

## Model Comparison

The initial models produced the following results:

| Model               | Accuracy | Bad Precision | Bad Recall |    Bad F1 |   ROC-AUC |
| ------------------- | -------: | ------------: | ---------: | --------: | --------: |
| Logistic Regression |    78.0% |         64.8% |  **58.3%** | **61.4%** | **83.4%** |
| Decision Tree       |    68.0% |         46.8% |      48.3% |     47.5% |     62.4% |
| Random Forest       |    75.5% |         63.4% |      43.3% |     51.5% |     81.7% |

### Model Selection

**Logistic Regression** demonstrated the strongest overall performance among the three initial models.

It achieved:

* Highest Accuracy
* Highest Bad Recall
* Highest Bad F1-score
* Highest ROC-AUC

Based on these results, Logistic Regression was selected for further optimization.

---

## Classification Threshold Optimization

The default classification threshold was further analyzed to improve the model's ability to identify **Bad Credit Risk** customers.

Multiple thresholds were evaluated based on:

* Accuracy
* Bad Precision
* Bad Recall
* Bad F1-score

The threshold that provided the best Bad F1-score was:

**Optimal Threshold = 0.71**

At this threshold:

* **Bad Recall = 80.0%**
* **Bad Precision = 56.5%**
* **Bad F1-score = 66.2%**

This demonstrates the trade-off between overall predictive performance and the ability to identify higher-risk customers.

---

## Feature Explainability

Feature explainability was performed using the coefficients of the Logistic Regression model.

The preprocessing process generated **42 model features** after encoding categorical variables.

Model coefficients were examined to understand the direction and relative magnitude of each feature's contribution to the model prediction.

This provides additional interpretability and helps connect the machine learning output with credit risk characteristics.

---

## Hyperparameter Tuning

**GridSearchCV** was used to optimize the Logistic Regression model.

The following hyperparameters were evaluated:

* `C`
* `solver`

A **5-fold cross-validation** strategy was used with **ROC-AUC** as the optimization metric.

### Best Parameters

```text
C = 0.1
solver = lbfgs
```

### Best Cross-Validated ROC-AUC

**77.9%**

---

## Final Model Evaluation

The tuned Logistic Regression model was evaluated on the unseen test set.

### Final Model Performance

| Metric        |    Result |
| ------------- | --------: |
| Accuracy      | **79.5%** |
| Bad Precision | **69.4%** |
| Bad Recall    | **56.7%** |
| Bad F1-score  | **62.4%** |
| ROC-AUC       | **83.6%** |

The final tuned model achieved a **ROC-AUC of 83.6%**, demonstrating good discriminatory ability between Good and Bad credit risk classes.

---

## Final Risk Threshold

The optimized threshold of **0.71** was applied to the tuned Logistic Regression model to prioritize the detection of Bad Credit Risk customers.

### Performance at the Optimized Threshold

| Metric        |    Result |
| ------------- | --------: |
| Threshold     |  **0.71** |
| Bad Recall    | **80.0%** |
| Bad Precision | **56.5%** |
| Bad F1-score  | **66.2%** |

The threshold adjustment increases the model's ability to identify Bad customers, while introducing the expected trade-off with precision.

This reflects a practical credit risk modelling consideration: **the optimal decision threshold depends on the relative business cost of false positives and false negatives.**

---

## Credit Risk Interpretation

The final model converts predicted probabilities into:

* **0 = Bad Credit Risk**
* **1 = Good Credit Risk**

Using the optimized threshold of **0.71**, the model becomes more focused on identifying potentially high-risk borrowers.

This demonstrates the importance of going beyond a default 0.50 classification threshold when applying machine learning to credit risk problems.

---

## Key Skills Demonstrated

### Credit Risk

* Credit Risk Assessment
* Loan Default Prediction
* Risk Classification
* Risk-Based Decision Making
* Borrower Risk Segmentation
* Model Interpretation

### Machine Learning

* Binary Classification
* Logistic Regression
* Decision Tree
* Random Forest
* Train/Test Split
* Cross-Validation
* Hyperparameter Tuning
* Threshold Optimization
* Model Comparison
* Model Evaluation

### Data & Python

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Jupyter Notebook
* Google Colab
* Data Preprocessing
* Feature Engineering / Feature Preparation
* Feature Explainability

---

## Tools & Technologies

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Jupyter Notebook**
* **Google Colab**

---

## Project Outcome

This project demonstrates the application of machine learning to a practical **credit risk modelling problem**, combining predictive modelling with risk-focused evaluation and interpretation.

The analysis showed that **Logistic Regression** provided the strongest initial performance among the tested models. After hyperparameter tuning, the final model achieved a **ROC-AUC of 83.6%** on the unseen test set.

Further threshold optimization increased **Bad Credit Risk recall to 80.0%**, demonstrating how classification thresholds can be adjusted to align model predictions with credit risk priorities.

Overall, the project demonstrates an end-to-end approach to developing, evaluating, interpreting, and optimizing a machine learning model for **credit risk and loan default prediction**.
