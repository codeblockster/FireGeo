ROC_AUC - 

PR_AUC - precision recall under curve 
Recall _ fire - 
recall_ fire = ( true positive / ( true positive + false negative ) )

  The result obtained for the ffopm diffrent algo ( gradient xgboost algorithm):
 
   XGBoost:
     ROC_AUC: 0.9586
     PR_AUC: 0.1941
     Recall_Fire: 0.7356
     Precision_Fire: 0.1446
     F1_Score: 0.2417
     False_Alarm_Rate: 0.0490
     Specificity: 0.9510

   LightGBM:
     ROC_AUC: 0.9577
     PR_AUC: 0.1928
     Recall_Fire: 0.7255
     Precision_Fire: 0.1432
     F1_Score: 0.2391
     False_Alarm_Rate: 0.0489
     Specificity: 0.9511

   CatBoost:
     ROC_AUC: 0.9597
     PR_AUC: 0.2048
     Recall_Fire: 0.7605
     Precision_Fire: 0.1434
     F1_Score: 0.2413
     False_Alarm_Rate: 0.0511
     Specificity: 0.9489