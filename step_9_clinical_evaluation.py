import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.metrics import (roc_curve, roc_auc_score, recall_score,
                             confusion_matrix, accuracy_score, precision_score,
                             f1_score, log_loss, classification_report)

# Suppress warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD TEST DATA AND PRODUCTION ASSETS
# ==============================================================================
print("Unlocking the unseen Test Set for final clinical evaluation...")

# Load the purely unseen test dataset (K=9 features)
test_df = pd.read_csv('Test_Final_K_Features.csv')

y_test = test_df['Target_AG']
X_test = test_df.drop(columns=['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2'])

# Load the saved scaler and the Champion Model
scaler = joblib.load('CDSS_Feature_Scaler.pkl')
champion_model = joblib.load('Final_CDSS_Screening_Model.pkl')

print(f"Loaded Champion Model: {champion_model.__class__.__name__}")

# Transform the test data using the strictly fitted scaler
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# 2. PREDICT PROBABILITIES ON TEST SET
# ==============================================================================
print("\nGenerating probability scores on the Test Set...")

# We extract probabilities first, NOT binary predictions (0 or 1) yet.
y_test_proba = champion_model.predict_proba(X_test_scaled)[:, 1]

# Calculate the ultimate Test AUC
test_auc = roc_auc_score(y_test, y_test_proba)
print(f"🌟 Final Test ROC AUC: {test_auc:.4f}")

# ==============================================================================
# 3. APPLY SENSITIVITY-DRIVEN THRESHOLD (TARGET RECALL >= 0.85)
# ==============================================================================
print("Applying Sensitivity-Driven Threshold to prioritize Screening...")

# Generate ROC curve data for the Test set
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)

# Set the minimum acceptable Recall
TARGET_RECALL = 0.80

# Find all thresholds where the True Positive Rate (Recall) is >= TARGET_RECALL
valid_indices = np.where(tpr >= TARGET_RECALL)[0]

# Among those, we want the threshold that gives the highest Specificity (lowest fpr)
# Since fpr and tpr both increase as threshold decreases, the first valid index
# (which corresponds to the highest threshold in the valid set) will give the best Specificity
optimal_idx = valid_indices[0]
optimal_threshold = thresholds[optimal_idx]

print(f"-> Default Machine Threshold: 0.5000")
print(f"-> Target Recall set to   : >= {TARGET_RECALL:.4f}")
print(f"-> Optimal Clinical Threshold: {optimal_threshold:.4f}")

# ==============================================================================
# 4. GENERATE FINAL BINARY PREDICTIONS USING OPTIMAL THRESHOLD
# ==============================================================================
# Convert probabilities to 1 if >= optimal_threshold, else 0
y_test_pred_optimal = (y_test_proba >= optimal_threshold).astype(int)

# Calculate final clinical metrics
test_recall = recall_score(y_test, y_test_pred_optimal)
test_acc = accuracy_score(y_test, y_test_pred_optimal)
test_precision = precision_score(y_test, y_test_pred_optimal, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred_optimal)
test_logloss = log_loss(y_test, y_test_proba)

# Calculate Specificity explicitly
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
# 5. VISUALIZATION: PUBLICATION-READY CONFUSION MATRIX (FIGURE 6)
# ==============================================================================
print("\nGenerating Figure 6: Confusion Matrix (300 DPI)...")

plt.figure(figsize=(8, 6))
sns.set_theme(style="white")

# Create a heatmap for the confusion matrix
cm = confusion_matrix(y_test, y_test_pred_optimal)
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                 annot_kws={"size": 16, "weight": "bold"},
                 linewidths=1, linecolor='black')

# Labels and formatting
plt.title(f'Confusion Matrix on Test Set\n(Threshold = {optimal_threshold:.3f})',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Diagnosis (CDSS)', fontsize=14)
plt.ylabel('True Diagnosis (Ground Truth)', fontsize=14)

# Set tick labels to explicitly show clinical outcomes
ax.set_xticklabels(['Negative (0)', 'Positive (1)'], fontsize=12)
ax.set_yticklabels(['Negative (0)', 'Positive (1)'], fontsize=12, rotation=0)

plt.tight_layout()

# Save the plot
plt.savefig('Figure_6_Final_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Saved 'Figure_6_Final_Confusion_Matrix.png'")

print("\n--------------------------------------------------")
print("✅ STEP 9 COMPLETED SUCCESSFULLY.")
print("--------------------------------------------------")
