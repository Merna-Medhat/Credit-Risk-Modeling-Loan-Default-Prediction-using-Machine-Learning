# Credit Risk Modeling & Loan Default Prediction using Machine Learning

## Project Overview

This project develops an end-to-end **Machine Learning solution for Credit Risk Assessment and Loan Default Risk Classification** using the **German Credit dataset**.

The objective is to classify customers into **Good or Bad Credit Risk** based on borrower and credit-related characteristics.

The project combines **credit risk concepts with machine learning techniques**, covering the complete modelling lifecycle from data understanding and quality assessment to preprocessing, model development, evaluation, threshold optimization, explainability, hyperparameter tuning, and final risk interpretation.

---

## Business Objective

The objective is to develop a predictive credit risk model that can help identify customers with a higher likelihood of being classified as **Bad Credit Risk**.

From a credit risk perspective, particular attention is given to the model's ability to identify Bad customers, since failing to detect a high-risk borrower can contribute to increased credit losses.

The project demonstrates how machine learning can support:

* Data-driven credit risk assessment
* Risk classification and segmentation
* Credit decision-making
* Identification of higher-risk borrowers
* Consistent evaluation of borrower creditworthiness
* Risk-based classification threshold selection

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

## Data Understanding & Quality Assessment

The dataset was examined to understand its structure, feature characteristics, and data quality before model development.

The analysis included:

* Dataset dimensions and structure
* Sample records
* Data types
* Missing value analysis
* Duplicate record analysis
* Target variable distribution
* Feature cardinality and number of unique values

### Data Quality Results

| Data Quality Check | Result |
| ------------------ | -----: |
| Rows               |  1,000 |
| Input Features     |     20 |
| Missing Values     |      0 |
| Duplicate Records  |      0 |

---

## Feature Classification

The input variables were classified according to their data characteristics and business meaning to ensure that appropriate preprocessing techniques were applied.

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

The project follows an end-to-end **credit risk modelling workflow**:

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

A **stratified train/test split** was used to preserve the Good/Bad credit risk distribution across both datasets.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The test set was kept separate and used for final model evaluation on **unseen data**.

---

## Data Preprocessing

Different preprocessing techniques were applied according to feature type.

| Feature Type | Preprocessing Technique |
| ------------ | ----------------------- |
| Numerical    | `StandardScaler`        |
| Ordinal      | `OrdinalEncoder`        |
| Categorical  | `OneHotEncoder`         |

A **ColumnTransformer** was used to combine the preprocessing steps into a unified transformation process.

The preprocessing was fitted on the training data and then applied to the test data to maintain a proper machine learning workflow and avoid information leakage.

After preprocessing and encoding, the model contained **42 transformed features**.

---

## Machine Learning Models

Three classification models were developed and evaluated.

### 1. Logistic Regression

**Logistic Regression** was selected as the baseline model due to its interpretability and suitability for binary credit risk classification.

It also provides coefficients that can be examined to understand the directional contribution of model features.

### 2. Decision Tree

A **Decision Tree Classifier** was developed as a non-linear modelling alternative to evaluate whether tree-based decision rules could improve predictive performance.

### 3. Random Forest

A **Random Forest Classifier** was developed as an ensemble-based alternative combining multiple decision trees.

---

## Model Evaluation

Model performance was evaluated using multiple classification metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC

Particular attention was given to the **Bad Credit Risk class**.

### Why Bad Recall Matters

In credit risk modelling, identifying potentially Bad customers is particularly important.

**Bad Recall** measures the proportion of actual Bad customers that the model successfully identifies.

A higher Bad Recall can help reduce the number of high-risk borrowers incorrectly classified as Good, although improving recall may introduce a trade-off with precision and overall classification performance.

---

## Model Comparison

The initial models produced the following results on the test set using the default classification threshold:

| Model               |  Accuracy | Bad Precision | Bad Recall |    Bad F1 |   ROC-AUC |
| ------------------- | --------: | ------------: | ---------: | --------: | --------: |
| Logistic Regression | **78.0%** |     **64.8%** |  **58.3%** | **61.4%** | **83.4%** |
| Decision Tree       |     68.0% |         46.8% |      48.3% |     47.5% |     62.4% |
| Random Forest       |     75.5% |         63.4% |      43.3% |     51.5% |     81.7% |

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

Model performance was further analyzed by evaluating different classification probability thresholds rather than relying only on the default threshold.

Thresholds ranging from **0.10 to 0.99** were evaluated using:

* Accuracy
* Bad Precision
* Bad Recall
* Bad F1-score

The threshold producing the best **Bad F1-score** was selected:

### Optimal Threshold = 0.71

At this threshold:

| Metric        |    Result |
| ------------- | --------: |
| Bad Recall    | **80.0%** |
| Bad Precision | **56.5%** |
| Bad F1-score  | **66.2%** |

The threshold optimization demonstrates the trade-off between identifying more Bad customers and maintaining precision.

In a real credit risk environment, the appropriate threshold would ultimately depend on the business objective and the relative cost of different types of classification errors.

---

## Feature Explainability

Feature explainability was performed using the **Logistic Regression coefficients**.

The preprocessing process generated **42 model-ready features** after scaling and encoding.

The coefficients were extracted and analyzed to understand:

* The direction of the relationship between transformed features and the model prediction
* The relative magnitude of model coefficients
* Which transformed variables have stronger influence within the fitted Logistic Regression model

This provides an interpretable view of the model and helps connect machine learning outputs with credit risk characteristics.

> Note: coefficient interpretation should be considered in the context of the applied scaling, ordinal encoding, and one-hot encoding transformations.

---

## Hyperparameter Tuning

After selecting Logistic Regression, **GridSearchCV** was used to optimize the model's hyperparameters.

The following parameters were evaluated:

* `C`
* `solver`

A **5-fold cross-validation** strategy was applied using **ROC-AUC** as the optimization metric.

### Parameter Grid

```python
C = [0.01, 0.1, 1, 10, 100]
solver = ['liblinear', 'lbfgs']
```

### Best Parameters

```text
C = 0.1
solver = lbfgs
```

### Best Cross-Validated ROC-AUC

**77.9%**

The cross-validated score represents performance during model selection on the training data, while the final ROC-AUC below represents performance on the unseen test set.

---

## Final Model Evaluation

The tuned Logistic Regression model was evaluated on the **unseen test set**.

### Final Model Performance

| Metric        |    Result |
| ------------- | --------: |
| Accuracy      | **79.5%** |
| Bad Precision | **69.4%** |
| Bad Recall    | **56.7%** |
| Bad F1-score  | **62.4%** |
| ROC-AUC       | **83.6%** |

The tuned Logistic Regression model achieved a **ROC-AUC of 83.6%**, demonstrating good discriminatory ability between Good and Bad Credit Risk classes on the unseen test set.

---

## Final Risk Threshold

The optimized threshold of **0.71** was applied to the tuned Logistic Regression model to place greater emphasis on identifying Bad Credit Risk customers.

### Performance at the Optimized Threshold

| Metric                   |    Result |
| ------------------------ | --------: |
| Classification Threshold |  **0.71** |
| Bad Recall               | **80.0%** |
| Bad Precision            | **56.5%** |
| Bad F1-score             | **66.2%** |

Compared with the tuned model at its default threshold, the optimized threshold substantially increases **Bad Recall from 56.7% to 80.0%**.

This comes with the expected precision trade-off and demonstrates an important credit risk modelling principle:

> **The optimal classification threshold should be aligned with the business cost of false positives and false negatives.**

---

## Credit Risk Interpretation

The model predicts the probability of a customer belonging to the **Good Credit Risk** class and converts the probability into a final classification.

The final classification is:

* **0 = Bad Credit Risk**
* **1 = Good Credit Risk**

Using the optimized threshold of **0.71**, the model is adjusted to become more conservative in identifying potentially higher-risk borrowers.

This demonstrates why model development in credit risk should not rely solely on overall accuracy. **Class-specific performance, probability thresholds, and business risk considerations** are also important when translating model predictions into credit decisions.

---

## Key Skills Demonstrated

### Credit Risk Analytics

* Credit Risk Assessment
* Loan Default Risk Classification
* Good/Bad Risk Classification
* Risk Segmentation
* Credit Decision Support
* Risk-Based Threshold Optimization
* Model Interpretation

### Machine Learning

* Binary Classification
* Logistic Regression
* Decision Tree
* Random Forest
* Train/Test Split
* Stratified Sampling
* Cross-Validation
* Hyperparameter Tuning
* GridSearchCV
* Threshold Optimization
* Model Comparison
* Model Evaluation

### Model Evaluation & Risk Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* Class-specific Performance Analysis
* Precision/Recall Trade-offs

### Data Preparation

* Data Quality Assessment
* Missing Value Analysis
* Duplicate Detection
* Feature Classification
* Numerical Feature Scaling
* Ordinal Encoding
* One-Hot Encoding
* ColumnTransformer
* Feature Preparation

### Python & Analytics Tools

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Jupyter Notebook
* Google Colab

---

## Tools & Technologies

| Category                    | Tools                          |
| --------------------------- | ------------------------------ |
| Programming                 | Python                         |
| Data Analysis               | Pandas, NumPy                  |
| Machine Learning            | Scikit-learn                   |
| Data Visualization          | Matplotlib                     |
| Development Environment     | Google Colab, Jupyter Notebook |
| Version Control & Portfolio | GitHub                         |

### Key Scikit-learn Components

* `train_test_split`
* `ColumnTransformer`
* `StandardScaler`
* `OrdinalEncoder`
* `OneHotEncoder`
* `LogisticRegression`
* `DecisionTreeClassifier`
* `RandomForestClassifier`
* `GridSearchCV`
* Classification metrics
* ROC-AUC evaluation

---

## Key Project Takeaways

The project demonstrates that **model performance should be evaluated from both a predictive and credit risk perspective**.

Key findings include:

1. **Logistic Regression** achieved the strongest initial performance among the tested models.
2. The tuned Logistic Regression model achieved **79.5% Accuracy** and **83.6% ROC-AUC** on the unseen test set.
3. **Bad Recall** was explicitly considered because correctly identifying higher-risk customers is an important credit risk objective.
4. Threshold optimization at **0.71** increased Bad Recall from **56.7% to 80.0%**.
5. The improvement in Bad Recall was accompanied by a precision trade-off, illustrating the importance of balancing model performance with business risk objectives.
6. Logistic Regression coefficients provided an interpretable view of feature contributions.

---

## Project Outcome

This project demonstrates the application of machine learning to a practical **credit risk modelling problem**, combining predictive modelling with risk-focused evaluation, threshold optimization, and model explainability.

The final workflow covers the complete modelling process:

**Data Understanding → Data Quality → Feature Preparation → Preprocessing → Model Development → Model Evaluation → Model Comparison → Threshold Optimization → Explainability → Hyperparameter Tuning → Final Evaluation → Credit Risk Interpretation**

The project highlights how machine learning can be used as a **decision-support tool within credit risk**, while recognizing that model thresholds and performance priorities should ultimately be aligned with business objectives and risk appetite.

---

## Repository Structure

```text
Credit-Risk-Machine-Learning-Loan-Default-Prediction/
│
├── Credit_Risk_Loan_Default_Prediction.ipynb
└── README.md
```

---

## Author

**Merna Medhat**

Credit Risk | Risk Analytics | Machine Learning | Data Analysis
