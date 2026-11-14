# Healthcare Insurance Fraud Detection Using Machine Learning

## Project Overview

Healthcare insurance fraud is a major challenge for insurance companies because fraudulent claims can result in financial losses and inefficient use of resources.

This project uses Machine Learning to identify potentially fraudulent healthcare insurance claims. The project follows a complete machine learning workflow, including data exploration, preprocessing, model building, cross-validation, hyperparameter tuning, model interpretation, error analysis, and final model saving.

The main objective is to build a classification model that can identify fraudulent claims while giving special importance to detecting as many fraudulent claims as possible.

---

## Problem Statement

The objective of this project is to develop a Machine Learning classification model that predicts whether a healthcare insurance claim is **fraudulent or legitimate**.

Since fraudulent claims represent a smaller portion of the dataset, this is an **imbalanced classification problem**. Therefore, accuracy alone is not sufficient for evaluating the model.

---

## Dataset

The dataset contains **10,000 healthcare insurance claim records** and **21 features**.

The dataset includes information related to:

- Patient details
- Provider information
- Diagnosis and procedure codes
- Claim amounts
- Insurance type
- Claim submission details
- Provider claim frequency
- Patient visits
- Length of stay
- Visit type
- Claim status
- Fraud indicators

### Target Variable

The target variable is:

`Fraud_Label`

It contains two classes:

- `Legitimate`
- `Fraud`

The dataset contains:

- Legitimate claims: **9,171 (91.71%)**
- Fraudulent claims: **829 (8.29%)**

Therefore, the dataset is imbalanced.

---

## Project Workflow

The project follows these major steps:

1. Problem Definition
2. Data Loading and Initial Inspection
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing
5. Feature Engineering
6. Model Building
7. Cross-Validation
8. Hyperparameter Tuning and Evaluation
9. Model Interpretation using Permutation Feature Importance
10. Error Analysis
11. Final Model Saving

---

## Exploratory Data Analysis

The dataset was explored to understand its structure, data types, missing values, duplicates, target distribution, and relationships between variables.

### Missing Values

Missing values were identified in:

- `Insurance_Type`
- `Provider_Specialty`
- `Prior_Visits_12m`

Each of these columns contained 350 missing values.

### Duplicate Check

No duplicate records were found in the dataset.

### Target Distribution

The target variable was highly imbalanced, with legitimate claims representing the majority of the records and fraudulent claims representing a smaller portion.

### Claim Amount Distribution

The distribution of `Claim_Amount` was positively skewed.

Most claim amounts were concentrated approximately between 200 and 800, while a smaller number of claims had much higher values.

This distribution was considered during the exploratory analysis because unusually high claim amounts may contain useful information for fraud prediction.

---

## Data Preprocessing

The following preprocessing steps were performed:

- Missing values were handled.
- Duplicate records were checked.
- Numerical and categorical features were identified.
- Categorical variables were encoded.
- Numerical features were scaled where required.
- The data was divided into training and testing sets.
- Preprocessing components were fitted using the training data to avoid data leakage.

The preprocessing components were saved along with the final model so that the same transformations can be applied to future data.

---

## Model Building

Several classification algorithms were considered for the fraud detection task.

The models included:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

The models were evaluated using metrics suitable for an imbalanced classification problem.

---

## Evaluation Metrics

Since this is an imbalanced classification problem, accuracy alone is not sufficient to evaluate model performance.

The following metrics were used:

### Precision

Precision measures how many claims predicted as fraudulent were actually fraudulent.

### Recall

Recall measures how many of the actual fraudulent claims were correctly detected.

Recall was given particular importance because missing a fraudulent claim can be costly.

### F1-Score

F1-score provides a balance between precision and recall.

It was selected as the primary scoring metric during hyperparameter tuning.

### ROC-AUC

ROC-AUC measures the model's ability to distinguish between fraudulent and legitimate claims.

### Confusion Matrix

The confusion matrix was used to examine correct and incorrect predictions, including false positives and false negatives.

---

## Cross-Validation

Cross-validation was performed to evaluate the stability of the Gradient Boosting model across different subsets of the training data.

The following metrics were evaluated:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Cross-validation helped provide a more reliable estimate of model performance than relying on a single training-validation split.

---

## Hyperparameter Tuning

Randomized Search with Cross-Validation was used to tune the Gradient Boosting model.

The tuning process used **F1-score** as the scoring metric because the dataset is imbalanced and both precision and recall are important.

The selected Gradient Boosting model used:

- `learning_rate = 0.05`
- `max_depth = 4`
- `n_estimators = 150`
- `random_state = 42`

---

## Threshold Tuning

The default classification threshold of 0.5 was adjusted to improve the detection of fraudulent claims.

A validation set was used to select a threshold targeting approximately 80% recall.

The selected threshold was:

**0.2216527**

This threshold was then applied to the test-set probabilities.

Threshold tuning improved the model's ability to identify fraudulent claims without changing the trained model's hyperparameters.

---

## Final Model Performance

Using the threshold-tuned Gradient Boosting model on the test set:

| Metric | Score |
|---|---:|
| Accuracy | 93.50% |
| Precision | 57.83% |
| Recall | 80.12% |
| F1-Score | 67.17% |
| ROC-AUC | 97.05% |

### Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 1737 | 97 |
| Actual Fraud | 33 | 133 |

The model correctly identified **133 out of 166 fraudulent claims**, achieving a recall of **80.12%**.

However, 33 fraudulent claims were missed and 97 legitimate claims were incorrectly flagged as fraudulent.

This shows the trade-off between detecting more fraudulent claims and reducing false fraud alerts.

---

## Overfitting Check

The threshold-tuned Gradient Boosting model achieved:

- Training accuracy: **94.09%**
- Testing accuracy: **93.50%**

The training and testing precision, recall, and F1-score were also reasonably close, with relatively small differences between training and testing performance.

Therefore, the model does not show significant signs of overfitting and demonstrates reasonable generalization to unseen test data.

---

## Model Interpretation

Permutation Feature Importance was used to understand which features contributed most to the tuned Gradient Boosting model's predictions.

The top features were:

| Feature | Importance |
|---|---:|
| Days_Between_Service_and_Claim | 0.496235 |
| Claim_Amount | 0.122874 |
| Number_of_Claims_Per_Provider_Monthly | 0.040171 |
| Length_of_Stay | 0.006402 |
| Diagnosis_Code_K21.9 | 0.003702 |

`Days_Between_Service_and_Claim` was the most influential feature, followed by `Claim_Amount` and `Number_of_Claims_Per_Provider_Monthly`.

Permutation importance indicates predictive contribution and does not establish that a feature directly causes healthcare insurance fraud.

---

## Error Analysis

The final threshold-tuned model produced the following results:

- True Positive (TP): 133
- True Negative (TN): 1737
- False Positive (FP): 97
- False Negative (FN): 33

### Observations

- 133 fraudulent claims were correctly identified.
- 1,737 legitimate claims were correctly classified.
- 97 legitimate claims were incorrectly classified as fraudulent.
- 33 fraudulent claims were not detected.

The results demonstrate the trade-off between increasing fraud detection and reducing false fraud alerts.

---

## Final Model Saving

The final trained model and preprocessing components were saved using `joblib`.

The saved model artifact contains:

- Trained Gradient Boosting model
- Selected classification threshold
- Encoder
- Scaler
- Numerical feature information
- Categorical feature information
- Feature column information
- Imputation values

The saved artifact can be used later for making predictions on new healthcare insurance claim data.

---

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## Project Structure

```text
Healthcare-Insurance-Fraud-Detection/
│
├── Healthcare_Insurance_Fraud_Detection.ipynb
├── README.md
└── .gitignore
```

The trained model file is generated locally during the project workflow and is not included in the current GitHub repository.

---

## Key Insights

The project showed that:

- Healthcare insurance fraud detection is an imbalanced classification problem.
- Accuracy alone is not sufficient for evaluating fraud detection models.
- Recall is particularly important because missing fraudulent claims can be costly.
- Gradient Boosting provided strong predictive performance for this dataset.
- Threshold tuning increased fraud detection recall to approximately 80%.
- `Days_Between_Service_and_Claim` was the most influential feature according to permutation importance.
- Error analysis showed that the model still produces both false positives and false negatives, highlighting the practical trade-off involved in fraud detection.

---

## Conclusion

This project demonstrates an end-to-end Machine Learning approach for healthcare insurance fraud detection.

The final threshold-tuned Gradient Boosting model achieved **80.12% recall** and **97.05% ROC-AUC** on the test set.

The project also included model interpretation and error analysis to understand how the model makes predictions and where it makes mistakes.

Overall, the project provides a practical framework for identifying potentially fraudulent healthcare insurance claims using Machine Learning.

