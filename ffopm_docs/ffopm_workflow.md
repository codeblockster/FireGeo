1 .Baseline Model 
train - 2000-2018
validate - 2018 - 2020
test - 2020 - 2025

algorithm used - logistic regression

Train / Validation / Test Data Stats:
Train: 538,069 rows, 3,241 fire events (0.60%)
Validation: 57,591 rows, 461 fire events (0.80%)
Test: 115,577 rows, 1,286 fire events (1.11%)

Performance Metrics (Test Set):
ROC-AUC: 0.845
PR-AUC: 0.094
Recall (Fire): 0.628
Precision (Fire): 0.075
F1-Score: 0.134
False Alarm Rate: 0.087
Specificity: 0.913

2. Advanced Model training : 
algorithms : XGBoost , 