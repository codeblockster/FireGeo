import json
import pandas as pd
import os

# Output folder
output_folder = "/Users/prabhatrawal/Minor_project_code/data/integrated_data/tuning_model_comparison"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "model_comparison_tuned.csv")

# Paths to metrics files
metrics_files = {
    "LightGBM": "/Users/prabhatrawal/Minor_project_code/data/integrated_data/lightgbm_tuning/tuning_metrics.json",
    "CatBoost": "/Users/prabhatrawal/Minor_project_code/data/integrated_data/catboost_tuning/s_tier_model_metrics.json",
    "XGBoost": "/Users/prabhatrawal/Minor_project_code/data/integrated_data/xgboost_tuning/enhanced_model_metrics.json"
}

def extract_metrics(model_name, file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    # LightGBM
    if model_name == "LightGBM":
        val = data.get("validation_metrics", {})
        test = data.get("test_metrics", {})

    # CatBoost
    elif model_name == "CatBoost":
        val = data.get("validation", {})
        test = data.get("test_optimal_threshold", {})

    # XGBoost
    elif model_name == "XGBoost":
        val = data.get("validation", {})
        test = data.get("test_optimal_threshold", {})

    rows = []
    # Validation row
    rows.append({
        "Model": model_name,
        "Split": "Validation",
        "ROC_AUC": val.get("ROC_AUC"),
        "PR_AUC": val.get("PR_AUC"),
        "Recall_Fire": val.get("Recall") or val.get("Recall_Fire"),
        "Precision_Fire": val.get("Precision") or val.get("Precision_Fire"),
        "F1_Score": val.get("F1_Score"),
        "False_Alarm_Rate": val.get("False_Alarm_Rate"),
        "Specificity": val.get("Specificity")
    })
    # Test row
    rows.append({
        "Model": model_name,
        "Split": "Test",
        "ROC_AUC": test.get("ROC_AUC"),
        "PR_AUC": test.get("PR_AUC"),
        "Recall_Fire": test.get("Recall") or test.get("Recall_Fire"),
        "Precision_Fire": test.get("Precision") or test.get("Precision_Fire"),
        "F1_Score": test.get("F1_Score"),
        "False_Alarm_Rate": test.get("False_Alarm_Rate"),
        "Specificity": test.get("Specificity")
    })
    return rows

# Build full table
all_rows = []
for model, path in metrics_files.items():
    all_rows.extend(extract_metrics(model, path))

df = pd.DataFrame(all_rows)
df = df[["Model", "Split", "ROC_AUC", "PR_AUC", "Recall_Fire", "Precision_Fire",
         "F1_Score", "False_Alarm_Rate", "Specificity"]]

# Save CSV
df.to_csv(output_file, index=False)
print(f"Comparison table saved to: {output_file}")
print(df)
