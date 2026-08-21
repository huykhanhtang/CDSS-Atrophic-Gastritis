import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, confusion_matrix
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD DATA, SCALER, AND CHAMPION MODEL
# ==============================================================================
print("Loading all datasets, scaler, and the Champion Ensemble Model...")

datasets = {
    'Train': pd.read_csv('CDSS_Web_App/Train_Final_K_Features.csv'),
    'Validation': pd.read_csv('Val_Final_K_Features.csv'),
    'Test': pd.read_csv('Test_Final_K_Features.csv')
}

scaler = joblib.load('CDSS_Feature_Scaler.pkl')
model = joblib.load('Final_CDSS_Screening_Model.pkl')


# ==============================================================================
# 2. BOOTSTRAPPING FUNCTION FOR 95% CONFIDENCE INTERVALS
# ==============================================================================
def calculate_metrics_with_ci(y_true, y_pred, y_proba, n_bootstraps=1000, ci=95):
    """
    Runs bootstrap resampling to calculate 95% CI for AUC, Recall, and Specificity.
    """
    bootstrapped_aucs = []
    bootstrapped_recalls = []
    bootstrapped_specs = []

    rng = np.random.RandomState(42)

    for i in range(n_bootstraps):
        # Bootstrap by sampling with replacement
        indices = rng.randint(0, len(y_true), len(y_true))

        if len(np.unique(y_true.iloc[indices])) < 2:
            # Skip if bootstrap sample only has one class
            continue

        y_true_b = y_true.iloc[indices]
        y_pred_b = y_pred[indices]
        y_proba_b = y_proba[indices]

        # AUC
        bootstrapped_aucs.append(roc_auc_score(y_true_b, y_proba_b))
        # Recall
        bootstrapped_recalls.append(recall_score(y_true_b, y_pred_b))
        # Specificity
        tn, fp, fn, tp = confusion_matrix(y_true_b, y_pred_b).ravel()
        bootstrapped_specs.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

    # Calculate lower and upper percentiles
    alpha = (100 - ci) / 2.0

    metrics = {
        'AUC': (np.mean(bootstrapped_aucs),
                np.percentile(bootstrapped_aucs, alpha),
                np.percentile(bootstrapped_aucs, 100 - alpha)),
        'Recall': (np.mean(bootstrapped_recalls),
                   np.percentile(bootstrapped_recalls, alpha),
                   np.percentile(bootstrapped_recalls, 100 - alpha)),
        'Specificity': (np.mean(bootstrapped_specs),
                        np.percentile(bootstrapped_specs, alpha),
                        np.percentile(bootstrapped_specs, 100 - alpha))
    }
    return metrics


# ==============================================================================
# 3. EVALUATE ACROSS ALL THREE SETS
# ==============================================================================
print("\nCalculating metrics and 95% CIs (Bootstrapping 1000 iterations)...")
print("This may take a minute...\n")

# NOTE: Use the Optimal Threshold founded in Step 9 here.
# Assuming it was roughly 0.40 (Replace this with your exact printed value from Step 9 if different)
OPTIMAL_THRESHOLD = 0.4081

table_data = []
roc_plot_data = {}

for split_name, df in datasets.items():
    y = df['Target_AG']
    X = df.drop(columns=['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2'])

    X_scaled = scaler.transform(X)

    y_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_proba >= OPTIMAL_THRESHOLD).astype(int)

    # Store ROC Data for plotting
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_plot_data[split_name] = {'fpr': fpr, 'tpr': tpr}

    # Calculate metrics with 95% CI
    results = calculate_metrics_with_ci(y, y_pred, y_proba)

    # Format strings for the table (e.g., "0.935 (0.910 - 0.950)")
    auc_str = f"{results['AUC'][0]:.3f} ({results['AUC'][1]:.3f}-{results['AUC'][2]:.3f})"
    recall_str = f"{results['Recall'][0]:.3f} ({results['Recall'][1]:.3f}-{results['Recall'][2]:.3f})"
    spec_str = f"{results['Specificity'][0]:.3f} ({results['Specificity'][1]:.3f}-{results['Specificity'][2]:.3f})"

    table_data.append({
        'Dataset': split_name,
        'N (Patients)': len(y),
        'AUC (95% CI)': auc_str,
        'Sensitivity / Recall (95% CI)': recall_str,
        'Specificity (95% CI)': spec_str
    })

# ==============================================================================
# 4. EXPORT MASTER CLINICAL TABLE (TABLE 2)
# ==============================================================================
master_table_df = pd.DataFrame(table_data)

print("=================================================================")
print("🏆 TABLE 2: MASTER CLINICAL PERFORMANCE (WITH 95% CI)")
print("=================================================================")
print(master_table_df.to_string(index=False))
print("=================================================================")

master_table_df.to_csv('Table_2_Master_Clinical_Performance.csv', index=False)
print("-> Saved 'Table_2_Master_Clinical_Performance.csv' (Ready for manuscript insertion)")

# ==============================================================================
# 5. VISUALIZATION: PUBLICATION-READY MULTI-SET ROC CURVE (FIGURE 7)
# ==============================================================================
print("\nGenerating Figure 7: ROC Curves for Train, Validation, and Test Sets...")

plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")

# Professional color palette for publication
colors = {'Train': '#2ca02c', 'Validation': '#1f77b4', 'Test': '#d62728'}
line_styles = {'Train': ':', 'Validation': '--', 'Test': '-'}
line_widths = {'Train': 2, 'Validation': 2.5, 'Test': 3}

for split_name, data in roc_plot_data.items():
    auc_val = float(master_table_df[master_table_df['Dataset'] == split_name]['AUC (95% CI)'].values[0].split(' ')[0])

    plt.plot(data['fpr'], data['tpr'],
             color=colors[split_name],
             linestyle=line_styles[split_name],
             linewidth=line_widths[split_name],
             label=f'{split_name} Set (AUC = {auc_val:.3f})')

# Random guess line
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
plt.title('Performance Consistency:\nROC Curves across Train, Val, and Test Sets',
          fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=12, frameon=True, shadow=True)
plt.tight_layout()

plt.savefig('Figure_7_MultiSet_ROC_Curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Saved 'Figure_7_MultiSet_ROC_Curve.png'")

print("\n--------------------------------------------------")
print("✅ STEP 11 COMPLETED SUCCESSFULLY.")
print("You now have the ultimate Table and Figure to prove your model's robustness!")
print("--------------------------------------------------")
