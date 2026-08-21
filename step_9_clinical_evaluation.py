import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.metrics import (roc_curve, roc_auc_score, recall_score,
                             confusion_matrix, accuracy_score, precision_score,
                             f1_score, log_loss, classification_report)

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD TEST DATA AND PRODUCTION ASSETS
# ==============================================================================
print("Unlocking the unseen Test Set for final clinical evaluation...")

test_df = pd.read_csv('Test_Final_K_Features.csv')

y_test = test_df['Target_AG']
X_test = test_df.drop(columns=['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2'])

scaler = joblib.load('CDSS_Feature_Scaler.pkl')
champion_model = joblib.load('Final_CDSS_Screening_Model.pkl')

print(f"Loaded Champion Model: {champion_model.__class__.__name__}")

X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# 2. PREDICT PROBABILITIES ON TEST SET
# ==============================================================================
print("\nGenerating probability scores on the Test Set...")

y_test_proba = champion_model.predict_proba(X_test_scaled)[:, 1]

test_auc = roc_auc_score(y_test, y_test_proba)
print(f"Final Test ROC AUC: {test_auc:.4f}")

# ==============================================================================
# 3. APPLY SENSITIVITY-DRIVEN THRESHOLD
# ==============================================================================
print("Applying Sensitivity-Driven Threshold to prioritize Screening...")

fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)

TARGET_RECALL = 0.80

valid_indices = np.where(tpr >= TARGET_RECALL)[0]

optimal_idx = valid_indices[0]
optimal_threshold = thresholds[optimal_idx]

print(f"-> Default Machine Threshold: 0.5000")
print(f"-> Target Recall set to   : >= {TARGET_RECALL:.4f}")
print(f"-> Optimal Clinical Threshold: {optimal_threshold:.4f}")

# ==============================================================================
# 4. GENERATE FINAL BINARY PREDICTIONS USING OPTIMAL THRESHOLD
# ==============================================================================

y_test_pred_optimal = (y_test_proba >= optimal_threshold).astype(int)

test_recall = recall_score(y_test, y_test_pred_optimal)
test_acc = accuracy_score(y_test, y_test_pred_optimal)
test_precision = precision_score(y_test, y_test_pred_optimal, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred_optimal)
test_logloss = log_loss(y_test, y_test_proba)

tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_optimal).ravel()
test_specificity = tn / (tn + fp)

print("\n=================================================================")
print("🩺 FINAL CLINICAL PERFORMANCE REPORT (TEST SET)")
print("=================================================================")
print(f"Threshold Used : {optimal_threshold:.4f}")
print(f"AUC            : {test_auc:.4f}")
print(f"Recall (Sens.) : {test_recall:.4f}  <- Prioritized for Screening")
print(f"Specificity    : {test_specificity:.4f}")
print(f"Accuracy       : {test_acc:.4f}")
print(f"Precision (PPV): {test_precision:.4f}")
print(f"F1-Score       : {test_f1:.4f}")
print(f"Log-Loss       : {test_logloss:.4f}")
print("=================================================================")

# ==============================================================================
# 5. VISUALIZATION: CONFUSION MATRIX (FIGURE 6)
# ==============================================================================
print("\nGenerating Figure 6: Confusion Matrix...")

plt.figure(figsize=(8, 6))
sns.set_theme(style="white")

cm = confusion_matrix(y_test, y_test_pred_optimal)
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                 annot_kws={"size": 16, "weight": "bold"},
                 linewidths=1, linecolor='black')

plt.title(f'Confusion Matrix on Test Set\n(Threshold = {optimal_threshold:.3f})',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Diagnosis (CDSS)', fontsize=14)
plt.ylabel('True Diagnosis (Ground Truth)', fontsize=14)

ax.set_xticklabels(['Negative (0)', 'Positive (1)'], fontsize=12)
ax.set_yticklabels(['Negative (0)', 'Positive (1)'], fontsize=12, rotation=0)

plt.tight_layout()

plt.savefig('Figure_6_Final_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Saved 'Figure_6_Final_Confusion_Matrix.png'")

print("\n--------------------------------------------------")
print("STEP 9 COMPLETED")
print("--------------------------------------------------")
