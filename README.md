# Early Diabetes Risk Prediction

## About This Project

This project aims to support early diabetes intervention by predicting diabetes risk from CDC behavioral survey data and identifying the key drivers behind that risk.

The project uses a LightGBM binary classification pipeline built in Python on a dataset with significant class imbalance, where only about 14% of patients had diabetes. To improve prediction quality for the minority class, SMOTE was applied to oversample diabetic cases during training.

## Project Goal

The main goal of this project is to build an interpretable machine learning model that can flag individuals at higher risk of diabetes using non-invasive behavioral, demographic, and lifestyle survey features.

Rather than relying only on model accuracy, the project focuses on recall-oriented evaluation because missing a high-risk patient can be more harmful than incorrectly flagging a low-risk patient for follow-up.

## Dataset

The analysis uses CDC behavioral health survey data related to diabetes risk indicators. The dataset includes features connected to health behavior, demographics, access to care, and self-reported health status.

The target variable is binary:

- `0`: No diabetes
- `1`: Diabetes

Because only about 14% of the dataset represents diabetic patients, class imbalance was a major modeling challenge.

## Methods

The project includes the following steps:

- Data cleaning and preprocessing
- Feature preparation for binary classification
- Handling class imbalance using SMOTE
- Training a LightGBM classification model
- Hyperparameter tuning to improve model performance
- Threshold optimization using Youden's J statistic
- Model interpretation using SHAP TreeExplainer

## Model

LightGBM was selected because it performs well on structured/tabular datasets and can capture non-linear relationships between health indicators and diabetes risk.

After hyperparameter tuning, the classification threshold was adjusted using Youden's J statistic to better balance sensitivity and specificity. This was especially important because the default 0.5 threshold was not ideal for the imbalanced dataset.

## Interpretability

SHAP TreeExplainer was used to explain the model's predictions at both the population and individual level.

The SHAP analysis helped identify the most influential risk factors driving diabetes predictions, making the model more transparent and useful for healthcare-related interpretation.

## Key Results

Despite the challenging class distribution, the model performed roughly three times better than a random classifier. This suggests that behavioral and demographic survey data can provide meaningful signals for early diabetes risk detection.

The project also shows that interpretable machine learning can help explain not only whether someone is at higher risk, but also which factors are contributing most to that risk.

## Methods/Libraries Used

- LightGBM
- SMOTE
- SHAP TreeExplainer
- Scikit-learn
- Pandas
- NumPy
- Matplotlib / Seaborn

## Why This Project Matters

Early diabetes detection can help connect high-risk individuals with clinical follow-up sooner. A low-cost screening model based on survey data could be especially useful in resource-limited healthcare settings where laboratory testing may be less accessible.

This project demonstrates how machine learning can support early intervention while still providing interpretable insights into the drivers of diabetes risk.

## Additional Files:


This repository also includes supporting files for reviewing the full project:

* **Final PDF Report** — A polished project report summarizing the objective, dataset, modeling strategy, performance results, fairness assessment, and final recommendations.
* **Python Source Code (`.ipynb`)** — The source analysis file containing the code used for data preparation, feature engineering, model training, evaluation, and reporting.
* **Markdown Report Version** — A GitHub-friendly version of the PDF report, including images and visual outputs, so the full analysis can be viewed directly in the repository.

These files provide both a high-level explanation of the project and the technical workflow behind the final results.


## Disclaimer

This project is for educational and research purposes only. It is not a medical diagnostic tool and should not replace professional clinical evaluation or laboratory testing.
