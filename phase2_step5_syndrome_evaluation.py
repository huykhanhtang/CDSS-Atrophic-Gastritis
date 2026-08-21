import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.metrics import (roc_auc_score, recall_score, precision_score,
                             f1_score, confusion_matrix, accuracy_score)

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD TEST DATA, SCALER, AND CHAMPION MODEL
# ==============================================================================
print("PHASE 2 - STEP 5: INDIVIDUAL SYNDROME CLINICAL EVALUATION")
print("Unlocking the unseen Test Set...")

test_df = pd.read_csv('Phase2_Test_Final_K_Features.csv')

target_tcm_cols = [col for col in test_df.columns if col.startswith('Target_TCM_')]
y_test = test_df[target_tcm_cols]
X_test = test_df.drop(columns=['Target_AG'] + target_tcm_cols)

# Tải Scaler và Mô hình Vô địch từ Step 4
scaler = joblib.load('Phase2_CDSS_Feature_Scaler.pkl')
champion_model = joblib.load('Phase2_Final_CDSS_MultiLabel_Model.pkl')

print(f"-> Loaded Champion Model successfully.")

X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# 2. PREDICT ON UNSEEN TEST SET
# ==============================================================================
print("\nGenerating Multi-label predictions on Test Set...")

# Lấy nhãn dự đoán (0/1)
y_test_pred = champion_model.predict(X_test_scaled)

# Lấy xác suất (Probabilities) để tính AUC
# Lưu ý: OneVsRestClassifier.predict_proba trả về ma trận (n_samples, n_classes)
y_test_proba = champion_model.predict_proba(X_test_scaled)

# ==============================================================================
# 3. EVALUATE EACH SYNDROME INDEPENDENTLY
# ==============================================================================
print("\nBreaking down performance for each of the 6 TCM Syndromes...")

syndrome_results = []

for i, col in enumerate(target_tcm_cols):
    syndrome_name = col.replace('Target_TCM_', '').replace('_', ' ')

    # Dữ liệu thực tế và dự đoán của riêng chứng hậu này
    y_true_single = y_test.iloc[:, i]
    y_pred_single = y_test_pred[:, i]
    y_proba_single = y_test_proba[:, i]

    # Tính toán các chỉ số Y khoa
    auc = roc_auc_score(y_true_single, y_proba_single)
    recall = recall_score(y_true_single, y_pred_single)
    precision = precision_score(y_true_single, y_pred_single, zero_division=0)
    f1 = f1_score(y_true_single, y_pred_single)
    acc = accuracy_score(y_true_single, y_pred_single)

    # Tính Specificity từ Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true_single, y_pred_single).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Đếm số ca mắc thực tế trong tập Test
    positive_cases = tp + fn

    syndrome_results.append({
        'TCM Syndrome': syndrome_name,
        'Positive Cases (n)': positive_cases,
        'AUC': auc,
        'Sensitivity (Recall)': recall,
        'Specificity': specificity,
        'Precision (PPV)': precision,
        'F1-Score': f1,
        'Accuracy': acc
    })

# ==============================================================================
# 4. EXPORT MASTER TABLE & VISUALIZATION
# ==============================================================================
results_df = pd.DataFrame(syndrome_results)

# Xuất ra file CSV (Đây chính là Table 3 trong bài báo của bạn)
results_df.to_csv('Phase2_Table_3_Individual_Syndromes_Performance.csv', index=False)

print("\n=======================================================================================================")
print("🩺 TABLE 3: CLINICAL PERFORMANCE BY INDIVIDUAL TCM SYNDROME (TEST SET)")
print("=======================================================================================================")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# Định dạng in số thập phân cho đẹp
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("=======================================================================================================")
print("-> Saved 'Phase2_Table_3_Individual_Syndromes_Performance.csv'")

# Vẽ biểu đồ Radar/Bar so sánh F1-Score giữa các chứng
plt.figure(figsize=(10, 6))
sns.barplot(x='F1-Score', y='TCM Syndrome', data=results_df.sort_values(by='F1-Score', ascending=False),
            palette='magma')
plt.title('Diagnostic F1-Score by TCM Syndrome (Test Set)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('F1-Score', fontsize=12)
plt.ylabel('')
plt.xlim(0, 1.05)

# Thêm nhãn số liệu lên biểu đồ
for index, value in enumerate(results_df.sort_values(by='F1-Score', ascending=False)['F1-Score']):
    plt.text(value + 0.01, index, f'{value:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('Phase2_Figure_5_Syndrome_F1_Scores.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Saved 'Phase2_Figure_5_Syndrome_F1_Scores.png'")
print("\n✅ PHASE 2 STEP 5 COMPLETED SUCCESSFULLY.")