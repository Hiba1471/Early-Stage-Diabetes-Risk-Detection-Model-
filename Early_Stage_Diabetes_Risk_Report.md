# Early Stage Diabetes Risk Detection Analysis
**A machine learning screening study using CDC health indicators**

**Author:** Hiba Khan  
**Date:** May 4, 2026  
**Focus:** non-invasive diabetes-risk prediction, recall-oriented evaluation, model interpretability  

- **253,680** instances
- **21** features
- **81%+** optimized recall

---

## Contents
1. [Introduction](#1-introduction)
2. [Data Explanation and Exploration](#2-data-explanation-and-exploration)
   - 2.1 [EDA and Assumption Checks](#eda-and-assumption-checks)
   - 2.2 [Preprocessing Strategy](#preprocessing-strategy)
3. [Algorithms and Techniques Analysis](#3-algorithms-and-techniques-analysis)
   - 3.1 [Hyperparameter Tuning and Class Imbalance](#hyperparameter-tuning-and-class-imbalance)
   - 3.2 [Initial Results at Default 0.5 Threshold](#initial-results-at-default-05-threshold)
   - 3.3 [Threshold Optimization with Youden's J Statistic](#threshold-optimization-with-youdens-j-statistic)
4. [Why the Ensemble Did Not Outperform](#4-why-the-ensemble-did-not-outperform)
5. [SHAP Explanations](#5-shap-explanations)
6. [Conclusion](#6-conclusion)

---

##  Introduction

Diabetes is one of the leading causes of immense financial pressure on international healthcare support systems, both individual and global. The prevalence of this chronic illness needs no introduction as cases keep surging with each passing year, up from 200 million in 1990 to 830 million in 2022. This is mostly caused by increasing obesity, sedentary careers, and more people aging than ever before. If we manage to detect this chronic public health crisis early, we can avoid or at least mitigate many of the severe complications like end-stage renal disease, cardiovascular disease, and neuropathy.

Clinical diagnosis has always been expensive, both financially and logistically, for under-insured and rural populations because it relies on laboratory tests. This is an even bigger problem in the third world, where even a lab itself might not exist to conduct these tests. These countries take the lion's share of diabetic cases by population percentage, going as far as approximately one in every third adult having diabetes in countries like Pakistan.

### Project Goal
This project investigates whether machine learning algorithms can accurately predict diabetes risk using only non-invasive, self-reported demographic and lifestyle data. If successful, the model could support a low-cost digital screening tool that flags high-risk individuals for clinical follow-up.

The analysis uses the CDC Diabetes Health Indicators Dataset, derived from the 2014 Behavioral Risk Factor Surveillance System (BRFSS). This telephone survey collected health and lifestyle data from roughly 400,000 respondents each year. Since BRFSS is self-reported, the dataset has limitations such as response bias, recall inaccuracy, and underreporting of stigmatized behaviors. The UCI Machine Learning Repository hosts the dataset and the original documentation.

Because missed diagnosis can have severe consequences, the model must put the highest priority on Recall while maintaining as much Specificity as possible. Interpretability is also important because opaque model outputs carry their own risks for time-pressed medical practitioners.

##  Data Explanation and Exploration

The dataset is comprised of 253,680 instances and 21 features, representing a large cross-section of the US population. The features span lifestyle and demographic domains, including BMI, physical activity, income brackets, age brackets, smoking status, self-rated health, and healthcare access. In clinical literature, these predictors are associated with metabolic disease.

The target variable is `Diabetes_binary`, where 0 means no diabetes and 1 means diabetes. The target distribution is highly imbalanced: 86.06% non-diabetic and 13.93% diabetic. Because the ratio is almost 6:1, preprocessing and metric selection had to account for class imbalance.

### EDA and Assumption Checks
Assumption checks were performed before constructing the models. Based on prior knowledge, multiple relationships were expected: inactive physical life correlates with increased BMI, which is expected to influence diabetes risk; similarly, reduced healthcare access is often associated with lower income.

To detect potential multicollinearity among features such as general health, mental health days, and BMI, A Pearson correlation matrix was generated and Variance Inflation Factors (VIF) were calculated for all predictors. The results showed that multicollinearity was not a limiting factor for linear models because VIF values stayed within acceptable bounds of 5.

### Preprocessing Strategy

**Preprocessing Pipeline**
1. **Categorical mapping and one-hot encoding:** Variables such as Age, General Health, Education, and Income were mapped to descriptive strings and then one-hot encoded so that each model could treat each category as a distinct input.
2. **Feature scaling:** Using StandardScaler, only the continuous BMI variable was standardized around a mean of 0 and a standard deviation of 1. Scaling was not applied to binary and one-hot encoded variables because those already exist on a $0/1$ scale.
3. **Stratified splitting:** Since only about 14% of the patients were diagnosed with diabetes, an $80/20$ train-test split stratified by outcome was used to preserve the same target distribution in training and testing sets.

##  Algorithms and Techniques Analysis

The machine learning techniques for this project include an ensemble approach using Gradient Boosting, Random Forest, and Logistic Regression. Logistic Regression was chosen as the main baseline because it is commonly used for binary classification and fits the CDC diabetes dataset, which is mostly binary or ordinal.

From a statistical perspective, Logistic Regression is appropriate because it models the probability that an individual will develop diabetes, making it a strong baseline for clinical use. It is also interpretable for healthcare practitioners, making it easier to determine which factors increase a patient's risk.

However, Logistic Regression alone is often not statistically powerful enough to capture the full complexity of the data. It assumes a linear relationship between the log odds of the risk factors and the target. Violations of assumptions, such as non-linearity or complex feature interactions, can lower its statistical power. For this reason, an ensemble-based approach using Gradient Boosting models such as XGBoost and LightGBM, along with Random Forest models, was expected to be more effective.

Gradient Boosting and Random Forest models can detect complex, high-order interactions between features without relying on strict statistical assumptions. Gradient Boosting improves performance by sequentially learning from previous errors and focusing heavily on observations that are difficult to classify.

### Hyperparameter Tuning and Class Imbalance
To combat the 6:1 class imbalance, algorithmic penalization  was implemented instead of synthetic resampling such as SMOTE. SMOTE added computational time and complexity without improving performance, especially since the dataset is large. Options such as `class_weight='balanced'` and `is_unbalance=True` were specified. Models were optimized with 3- to 5-fold GridSearchCV, using Recall as the primary scoring metric to prioritize catching true diabetes cases.

### Initial Results at Default 0.5 Threshold
At the default decision boundary of 0.5, the models showed erratic behavior due to aggressive class weighting.

**Table 1: Model evaluation metrics at the default threshold of 0.5.**

| Model | Sensitivity | Specificity | F1 |
| :--- | :--- | :--- | :--- |
| Logistic Regression | 0.775 | 0.718 | 0.441 |
| LightGBM | 0.166 | 0.978 | 0.439 |
| XGBoost | 0.944 | 0.390 | 0.330 |
| Ensemble (Soft Voting) | 0.3685 | 0.932 | 0.467 |

The Soft Voting Ensemble averaged the predicted probabilities of Logistic Regression, XGBoost, and LightGBM. Its weak recall score rejected the hypothesis that ensembling would automatically improve predictive performance.

### Threshold Optimization with Youden's J Statistic
Youden's J index was calculated to for threshold optimization, $J = Sensitivity + Specificity - 1$, across the ROC curve. This is a common healthcare evaluation approach and helped identify the optimal threshold for each model by balancing Recall and Specificity.

**Table 2: Model evaluation metrics at optimized Youden's J statistic thresholds.**

| Model | Optimized Threshold | Sensitivity | Specificity | F1 |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.4560 | 0.8169 | 0.6782 | 0.4295 |
| LightGBM | 0.4860 | 0.8100 | 0.6913 | 0.4363 |
| XGBoost | 0.1383 | 0.7868 | 0.7131 | 0.4422 |
| Ensemble | 0.3594 | 0.8116 | 0.6903 | 0.4358 |

**Threshold Finding** The default 0.5 boundary is a misleading cutoff point for this data because aggressive class weighting shifted model probabilities away from the midpoint. Optimized thresholds produced much more clinically useful results.

*(Figure 1: Calibration, Precision-Recall, and ROC curves for benchmark models.)*

LightGBM matches the Ensemble across major metrics while remaining better calibrated. The calibration plot shows that the models struggle with very high and very low probabilities. They are not confident enough when predicting low-risk cases and can be too confident when predicting high-risk cases. The probability scores do not perfectly reflect reality. However, a model can still correctly rank which individual is more at risk compared with others, even if it is not definitive.

A PR AUC of 0.40-0.42 conveys more information than ROC AUC when positive cases are low. It is mathematically about three times better than random chance because a random model would score roughly 0.139, matching the diabetic base rate of the dataset. Since this project prioritizes flagging as many true cases of diabetes as possible, F1 scores remain low. In a healthcare context, it is more acceptable to flag a healthy person for follow-up than to miss someone who is actually at risk.

## Why the Ensemble Did Not Outperform

The optimized benchmark models were on par with, and sometimes even outperformed, the ensemble. There are three major reasons why the added complexity of combining models failed to produce worthwhile improvements.

**Reason 1: Probability Calibration Divergence** Soft voting takes the arithmetic mean of each model's confidence scores. Logistic Regression and tree-based models produce confidence scores on very different scales. Logistic Regression outputs more balanced estimates, while LightGBM, forced to focus aggressively on the minority class, produced more skewed scores. Averaging mismatched outputs distorted the ensemble's decision boundary.

**Reason 2: Feature Space Redundancy** LightGBM and XGBoost are built on similar underlying ideas, so they learned very similar patterns from the data. Averaging two highly similar models did not meaningfully improve predictions; it mostly added redundant computation.

**Reason 3: The Equalizing Power of Youden's J** Ensembles often perform better because they smooth out variance across individual models. But once each single model is explicitly threshold-optimized with Youden's J, much of that advantage disappears. After finding a strong operational point for LightGBM and Logistic Regression, the marginal gain of ensembling approaches zero.

Following Occam's Razor, the LightGBM model is the superior practical choice. The ensemble uses around three times more memory and more training time just to reach similar results. LightGBM and Logistic Regression were close, but LightGBM was chosen because it can identify complex relationships that Logistic Regression cannot structurally capture. It also achieved better Specificity and F1 while supporting SHAP-based explanations, which make the tool more interpretable for clinicians.

## SHAP Explanations

SHAP (SHapley Additive exPlanations) values were generated using TreeExplainer on the LightGBM model. This identified which features influenced each individual prediction the most and made the model's predictions more readable and explainable.

*(Figure 2: Global SHAP plot showing the top 15 features driving LightGBM predictions.)*

The SHAP plot shows the features that most influenced the model's predictions. The three strongest indicators match clinical research about diabetes risk factors: high blood pressure, BMI, and self-rated general health. High cholesterol is also close behind. Interestingly, people rating their own health as excellent or very good were strongly predicted as low risk, which suggests that a patient's own perception of health is genuinely informative.

Income also ranks highly, appearing above several more clinical variables. This shows how financial stability may affect health outcomes downstream. Age also contributes to predictions across different brackets, though no single age group can be isolated as the sole driver.

*(Figure 3: Individual SHAP waterfall plot for Patient 33, a confirmed diabetic patient with a predicted diabetes probability of 69%.)*

The code was meant to find the first true positive case, compute individual SHAP values, and create a waterfall plot. The waterfall plot explains how the model reached a 69% predicted probability of diabetes for a single patient who later turned out to be confirmed diabetic. Each bar shows the effect of a specific feature in pushing or pulling the prediction toward a positive diagnosis.

For this patient, BMI made the largest contribution at +1.07, followed by high blood pressure (+0.61) and high cholesterol (+0.39). The patient also self-rated their health poorly (+0.32) and had a low income bracket (+0.11). This aligns with the model's broader emphasis on both clinical and socioeconomic indicators. The plot provides a concise, feature-level breakdown that a clinician could use to explain a patient's risk profile.

## Discussion

This project demonstrates that self-reported survey data alone can successfully flag individuals at high risk for diabetes, with over 81% Recall. Contrary to the initial hypothesis, the increased effort and computation of building a more complex ensemble model did not produce meaningful benefit. A single well-tuned LightGBM model turned out to be the most efficient and clinically practical solution for this use case.

That said, the model still has a clear ceiling. Survey-based data is inherently limited: people do not always introspect accurately or honestly report health behaviors. The diagnostic certainty of a blood test cannot be replicated with a simple questionnaire. Regardless of model tuning, this limits how precise the model can realistically be.

### Future Research Questions
- Would performance improve with biological markers such as glucose levels, insulin levels, HbA1c, and lipid profiles?
- Is the model unbiased across racial, socioeconomic, and geographic groups, or is it reflecting healthcare divides embedded in the data?
- Could NLP applied to medical notes or electronic health records outperform a structured survey-based model?
