# DL_MINI_PROJECT
# Student Academic Performance Prediction using a Deep Neural Network

A deep learning mini project that predicts whether a student is **at risk of failing** before the semester examinations, so the university can give early academic support.

## Problem Statement
Many students fail because their academic problems are noticed too late. This project builds a Deep Neural Network (TensorFlow/Keras) that uses attendance, assignment marks, internal scores, study hours, and previous GPA to predict a student's risk of failing.

## Dataset
- A **synthetic dataset** of 1,000 student records, generated with Python (NumPy and Pandas) and saved as `student_data.csv`.
- Features: `Attendance`, `Assignment_Marks`, `Internal_Score`, `Study_Hours`, `Previous_GPA`
- Target: `At_Risk` (1 = likely to fail, 0 = likely to pass)
- Class split: 678 Not At Risk, 322 At Risk

## Model Architecture
```
Input (5 features)
 -> Dense(64, ReLU) + BatchNorm + Dropout(0.3)
 -> Dense(32, ReLU) + BatchNorm + Dropout(0.2)
 -> Dense(16, ReLU)
 -> Dense(1, Sigmoid)
```
Trained with Adam (learning rate 0.001), binary cross-entropy, class weights, and Early Stopping.

## Results (test set of 200 students)
| Metric | Value |
|---|---|
| Accuracy | 76% |
| ROC-AUC | 0.876 |
| Recall (At Risk) | 0.78 |
| Precision (At Risk) | 0.59 |
| F1-score (At Risk) | 0.67 |

## Risk Levels
| Risk Probability | Level | Suggested Action |
|---|---|---|
| 70% or more | High | Immediate counselling and remedial classes |
| 40% to 70% | Medium | Mentor follow-up and study plan |
| Below 40% | Low | Regular monitoring |

## Tools Used
Python, Google Colab, TensorFlow, Keras, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib

## How to Run
1. Open `STUDENT_PERFORMANCE_PREDICTOR.ipynb` in Google Colab.
2. Run all cells from top to bottom. The first cell generates `student_data.csv`.
3. The notebook trains the model, shows the graphs and evaluation, and prints a sample student risk report.

## Limitations
- The dataset is synthetic, so the results reflect the patterns built into the data and may differ on real university data.
- Only five features are used. Factors like health and family situation are not captured.
- Predictions are probabilities and should support human decisions, not replace them.

## Future Scope
Explainable AI (SHAP/LIME), a web dashboard for faculty, automated alerts to mentors, and comparison with Random Forest and XGBoost.
