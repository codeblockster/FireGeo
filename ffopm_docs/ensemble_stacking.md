1. safety first approach
focus : catboost 
recall = 0.889 which means no missing of fire but high false alarm

2. resource efficiency
focus : lightgbm
You are missing almost half of the actual fires (Recall of 0.530).

# smart approach : 
catboost - 50%
lightgbm - 30%
xgboost - 20%

2️⃣ Algorithmic diversity

output : 
Average pairwise correlation: 0.3477
Diversity score: 0.6523

lightgbm: range 0.0000 - 0.0003
xgboost:  range 0.0000 - 0.0000


This is expected given:

extreme class imbalance

conservative tree models

cost-sensitive thresholding

rare-event prediction (fire days)

What matters more is:

PR-AUC

Recall at optimized threshold

Expected cost