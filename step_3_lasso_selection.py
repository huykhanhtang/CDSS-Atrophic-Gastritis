import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD NON-COLLINEAR DATA (FROM STEP 2)
# ==============================================================================
print("Loading datasets free of multicollinearity...")

train_df = pd.read_csv('Train_No_Collinear.csv')
val_df = pd.read_csv('Val_No_Collinear.csv')
test_df = pd.read_csv('Test_No_Collinear.csv')

y_train_ag = train_df['Target_AG']
# Đã cập nhật 2 cột TCM theo cấu trúc Multi-Label
X_train = train_df.drop(columns=['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2'])

print(f"Starting LASSO feature selection with {X_train.shape[1]} candidate features...")

# Ensure data is scaled before LASSO (Crucial for correct penalization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ==============================================================================
# 2. PERFORM LASSO REGRESSION WITH CROSS-VALIDATION
# ==============================================================================
print("Running L1-Regularized Logistic Regression (5-Fold CV)...")

n_alphas = 100
C_values = np.logspace(-3, 2, n_alphas)

lasso_cv = LogisticRegressionCV(
    Cs=C_values,
    cv=5,
    penalty='l1',
    solver='liblinear',
    scoring='neg_log_loss',
    random_state=42,
    max_iter=2000,
    n_jobs=-1
)

lasso_cv.fit(X_train_scaled, y_train_ag)

# ==============================================================================
# 3. COMPUTE THE COEFFICIENT PATH AND APPLY 1-SE RULE
# ==============================================================================
print("Computing coefficient paths and applying 1-SE Rule for stricter filtering...")

coefs = []
for c in C_values:
    lr = LogisticRegression(penalty='l1', solver='liblinear', C=c, random_state=42, max_iter=2000)
    lr.fit(X_train_scaled, y_train_ag)
    coefs.append(lr.coef_[0])

coefs = np.array(coefs)

log_lambdas = np.log(1 / C_values)

cv_scores = -lasso_cv.scores_[1]
mean_cv_dev = np.mean(cv_scores, axis=0)
std_cv_dev = np.std(cv_scores, axis=0) / np.sqrt(5)

# --- QUY TẮC 1-SE (1-Standard Error Rule) ---
# 1. Tìm điểm sai số thấp nhất (min)
min_idx = np.argmin(mean_cv_dev)
min_dev = mean_cv_dev[min_idx]
min_se = std_cv_dev[min_idx]

# 2. Tính ngưỡng 1-SE
threshold_1se = min_dev + min_se

# 3. Chọn Lambda lớn nhất (C nhỏ nhất -> Phạt mạnh nhất) mà sai số vẫn <= threshold_1se
valid_indices = np.where(mean_cv_dev <= threshold_1se)[0]
idx_1se = valid_indices[0]

opt_log_lambda_1se = log_lambdas[idx_1se]
optimal_C_1se = C_values[idx_1se]

print(f"-> Optimal C (Min Error): {C_values[min_idx]:.4f} (Too many features)")
print(f"-> Optimal C (1-SE Rule): {optimal_C_1se:.4f} (Stricter - Selected for final model)")

# ==============================================================================
# 4. PLOT 1: GLMNET-STYLE LASSO FIGURES (SIDE-BY-SIDE)
# ==============================================================================
print("Generating publication-ready LASSO plots (300 DPI)...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Left Plot: Cross-Validation Deviance ---
ax1.errorbar(log_lambdas, mean_cv_dev, yerr=std_cv_dev, fmt='o', color='#3498db',
             ecolor='lightgray', elinewidth=1, capsize=3, markersize=4)

# Vẽ đường Min (Xám đứt) và đường 1-SE (Đỏ nét đứt - Lựa chọn của chúng ta)
ax1.axvline(log_lambdas[min_idx], linestyle=':', color='gray', alpha=0.8, label='Min Deviance')
ax1.axvline(opt_log_lambda_1se, linestyle='--', color='#e74c3c', linewidth=2, label='1-SE Rule (Chosen)')

ax1.set_xlabel(r'Log($\lambda$)', fontsize=12)
ax1.set_ylabel('Binomial Deviance (Negative Log-Loss)', fontsize=12)
ax1.set_title('LASSO Cross-Validation Plot', fontsize=14)
ax1.legend()

# --- Right Plot: Coefficient Path ---
for i in range(coefs.shape[1]):
    ax2.plot(log_lambdas, coefs[:, i], lw=1.5, alpha=0.8)

ax2.axvline(opt_log_lambda_1se, linestyle='--', color='#e74c3c', linewidth=2, label='1-SE Rule Cutoff')
ax2.set_xlabel(r'Log($\lambda$)', fontsize=12)
ax2.set_ylabel('Coefficients', fontsize=12)
ax2.set_title('LASSO Coefficient Path', fontsize=14)
ax2.legend()

plt.tight_layout()
plt.savefig('Figure_2_LASSO_Path_and_CV.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Saved 'Figure_2_LASSO_Path_and_CV.png'")

# ==============================================================================
# 5. IDENTIFY SELECTED FEATURES & EXPORT CLEAN DATASETS
# ==============================================================================
# Trích xuất hệ số của điểm 1-SE thay vì điểm Min
lasso_coefs = coefs[idx_1se]
feature_names = X_train.columns

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lasso_coefs,
    'Abs_Coefficient': np.abs(lasso_coefs)
})

# THUẦN TÚY DỰA VÀO 1-SE RULE: Chỉ lọc bỏ những biến bị LASSO ép về đúng bằng 0
selected_features_df = coef_df[coef_df['Coefficient'] != 0].sort_values(by='Abs_Coefficient', ascending=False)
selected_features_list = selected_features_df['Feature'].tolist()

print(f"\n✅ LASSO (1-SE Rule) natively selected {len(selected_features_list)} crucial features out of {len(feature_names)}.")
print("Top Most Important Features:")
print(selected_features_df.head(15)[['Feature', 'Coefficient']].to_string(index=False))

# Export final datasets
columns_to_keep = selected_features_list + ['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2']
train_df[columns_to_keep].to_csv('Train_LASSO_Selected.csv', index=False)
val_df[columns_to_keep].to_csv('Val_LASSO_Selected.csv', index=False)
test_df[columns_to_keep].to_csv('Test_LASSO_Selected.csv', index=False)

print("\n--------------------------------------------------")
print("✅ STEP 3 COMPLETED SUCCESSFULLY.")
print("--------------------------------------------------")