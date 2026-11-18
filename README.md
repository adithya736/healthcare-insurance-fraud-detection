# Healthcare Insurance Fraud Detection Using Machine Learning

## Project Overview

Healthcare insurance fraud is a major challenge for insurance companies. Fraudulent claims can cause financial losses and increase the workload involved in claim verification.

This project uses Machine Learning to identify potentially fraudulent healthcare insurance claims. It follows an end-to-end machine learning workflow, including data exploration, preprocessing, model building, cross-validation, hyperparameter tuning, threshold tuning, model interpretation, error analysis, and model saving.

The main objective is to identify potentially fraudulent claims while giving particular importance to detecting as many fraudulent claims as possible.

## Project Links

- **GitHub Repository:** https://github.com/adithya736/healthcare-insurance-fraud-detection
- **Dataset Source:** https://www.kaggle.com/datasets/nudratabbas/healthcare-fraud-detection-dataset
- **Live Demo (AWS):** To be added after deployment.

## Problem Statement

The objective is to develop a Machine Learning classification model that predicts whether a healthcare insurance claim is fraudulent or legitimate.

Since fraudulent claims represent a smaller portion of the dataset, this is an imbalanced classification problem. Therefore, accuracy alone is not sufficient for evaluating model performance.

## Dataset

The project uses a healthcare insurance claims dataset downloaded from Kaggle.

- **Records:** 10,000
- **Features:** 21
- **Target variable:** `Is_Fraud`
- **0:** Legitimate
- **1:** Fraudulent

The dataset includes patient details, provider information, diagnosis and procedure codes, claim amounts, insurance type, claim submission details, provider claim frequency, and other claim-related information.

### Target Distribution

| Class | Records | Percentage |
|---|---:|---:|
| Legitimate | 9,171 | 91.71% |
| Fraudulent | 829 | 8.29% |

The dataset is imbalanced, with legitimate claims forming the majority class.

## Project Workflow

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

## Exploratory Data Analysis

The dataset was explored to understand its structure, data types, missing values, duplicates, target distribution, and feature distributions.

### Missing Values

Missing values were identified in:

- `Insurance_Type`
- `Provider_Specialty`
- `Prior_Visits_12m`

Each column contained 350 missing values.

### Duplicate Check

No duplicate records were found.

### Claim Amount Distribution

The `Claim_Amount` distribution was positively skewed. Most claim amounts were concentrated approximately between 200 and 800, while a smaller number of claims had higher values.

## Data Preprocessing

The following preprocessing steps were performed:

- Missing-value handling
- Numerical and categorical feature identification
- Categorical encoding
- Numerical scaling
- Training and testing data separation
- Prevention of data leakage by fitting preprocessing components on training data

The preprocessing components were saved along with the final model.

## Model Building

The following classification algorithms were considered:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Gradient Boosting was selected as the final model.

## Cross-Validation and Hyperparameter Tuning

Cross-validation was used to evaluate model stability across different subsets of the training data.

Randomized Search with Cross-Validation was used for hyperparameter tuning, with F1-score as the scoring metric.

Selected model parameters:

- `learning_rate = 0.05`
- `max_depth = 4`
- `n_estimators = 150`
- `random_state = 42`

## Threshold Tuning

The default classification threshold of 0.5 was adjusted to improve fraud detection.

The selected threshold was:

**0.2216527**

The threshold was selected using validation data and then applied to test-set probabilities.

## Final Model Performance

The final threshold-tuned Gradient Boosting model achieved the following test results:

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
| Actual Legitimate | 1,737 | 97 |
| Actual Fraud | 33 | 133 |

The model correctly identified 133 out of 166 fraudulent claims, achieving 80.12% recall.

## Model Interpretation

Permutation Feature Importance was used to understand which features contributed most to the model's predictions.

Top influential features:

| Feature | Importance |
|---|---:|
| Days_Between_Service_and_Claim | 0.496235 |
| Claim_Amount | 0.122874 |
| Number_of_Claims_Per_Provider_Monthly | 0.040171 |
| Length_of_Stay | 0.006402 |
| Diagnosis_Code_K21.9 | 0.003702 |

These values indicate predictive contribution, not direct causation.

## Error Analysis

The final model produced:

- True Positive: 133
- True Negative: 1,737
- False Positive: 97
- False Negative: 33

The results show the trade-off between detecting more fraudulent claims and reducing false fraud alerts. Predictions should be treated as indicators for further human review, not proof of fraud.

## Overfitting Check

- Training Accuracy: 94.09%
- Testing Accuracy: 93.50%

The relatively small difference between training and testing accuracy suggests reasonable generalization to unseen data.

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## Project Structure

```text
Healthcare-Insurance-Fraud-Detection/
│
├── Healthcare_Insurance_Fraud_Detection.ipynb
├── README.md
└── .gitignore
```

The trained model artifact is generated locally and is not currently included in the GitHub repository.

## Key Insights

- Healthcare insurance fraud detection is an imbalanced classification problem.
- Accuracy alone is not sufficient for evaluating fraud detection models.
- Recall is important because missing fraudulent claims can be costly.
- Gradient Boosting provided strong predictive performance.
- Threshold tuning increased fraud detection recall to 80.12%.
- `Days_Between_Service_and_Claim` was the most influential feature according to permutation importance.
- Error analysis highlighted both false positives and false negatives.

## Conclusion

This project demonstrates an end-to-end Machine Learning approach for healthcare insurance fraud detection.

The final threshold-tuned Gradient Boosting model achieved 80.12% recall and 97.05% ROC-AUC on the test set.

The project demonstrates how Machine Learning can support the identification of potentially fraudulent healthcare insurance claims and help prioritize claims for further investigation.

**Note:** The model supports claim screening and does not independently establish whether a claim is fraudulent.
