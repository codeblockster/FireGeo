1. ROC_AUC (overall classification ability):
CatBoost > LightGBM > XGBoost

2. PR_AUC (precision-recall, more important for imbalanced data like fire detection):
CatBoost > LightGBM > XGBoost

3. Recall_Fire (sensitivity to fire events):
CatBoost > XGBoost > LightGBM

4. Precision_Fire (how many predicted fires were correct):
LightGBM > XGBoost > CatBoost

5. F1_Score (balance of precision and recall):
LightGBM > XGBoost ≈ CatBoost

6. False_Alarm_Rate / Specificity (avoiding false positives):
LightGBM is best (FAR 0.0243, specificity 0.9757)
CatBoost has very high recall but a higher false alarm rate.