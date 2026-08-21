import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, f1_score
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD DATASETS FROM OVR LASSO
# ==============================================================================
print("Loading datasets from Phase 2 Step 1 for Multi-label RFECV Analysis...")

train_df = pd.read_csv('Phase2_Train_LASSO_Selected.csv')
val_df = pd.read_csv('Phase2_Val_LASSO_Selected.csv')
test_df = pd.read_csv('Phase2_Test_LASSO_Selected.csv')

target_tcm_cols = [col for col in train_df.columns if col.startswith('Target_TCM_')]
y_train = train_df[target_tcm_cols]

X_train = train_df.drop(columns=['Target_AG'] + target_tcm_cols)

available_features = X_train.shape[1]
print(f"Candidate features available from LASSO: {available_features}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ==============================================================================
# 2. CONFIGURE OVR RFECV FOR MULTI-LABEL CLASSIFICATION
# ==============================================================================
print("\nExecuting Multi-label RFECV using Macro F1-Score...")
print("Please wait, this evaluates subsets across 6 simultaneous disease targets...")

def ovr_importance_getter(fitted_estimator):
    
    coefs = np.array([est.coef_[0] for est in fitted_estimator.estimators_])

    mean_abs_coefs = np.mean(np.abs(coefs), axis=0)
    return mean_abs_coefs

base_estimator = LogisticRegression(C=1, solver='liblinear', class_weight='balanced', random_state=42, max_iter=2000)

ovr_estimator = OneVsRestClassifier(base_estimator)

from sklearn.model_selection import KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)

rfecv = RFECV(
    estimator=ovr_estimator,
    step=1,
    cv=cv,
    scoring='f1_macro',
    importance_getter=ovr_importance_getter,  # <--- ĐIỂM SỬA LỖI CHÍNH Ở ĐÂY
    min_features_to_select=5,
    n_jobs=-1
)

rfecv.fit(X_train_scaled, y_train)

optimal_k_math = rfecv.n_features_
print(f" RFECV Optimization successfully converged at K = {optimal_k_math} features.")

# ==============================================================================
# 3. EXTRACT VARIABLE LIST
# ==============================================================================
selected_features = X_train.columns[rfecv.support_].tolist()

# ==============================================================================
# 4. VISUALIZATION
# ==============================================================================
print("Generating Phase 2 RFECV Curve (Figure 2)...")

plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

cv_results = rfecv.cv_results_['mean_test_score']
min_features = rfecv.min_features_to_select
x_axis = range(min_features, min_features + len(cv_results))

plt.plot(x_axis, cv_results, marker='o', linestyle='-', color='#8e44ad', linewidth=2, markersize=5)

y_min = min(cv_results)
y_max = max(cv_results)
y_range = y_max - y_min
if y_range < 0.05:
    plt.ylim(y_max - 0.02, min(1.0, y_max + 0.02))
else:
    margin = y_range * 0.05
    plt.ylim(y_min - margin, min(1.0, y_max + margin))

y_offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.15

optimal_idx = optimal_k_math - min_features
optimal_score = cv_results[optimal_idx]

plt.axvline(x=optimal_k_math, color='#d7191c', linestyle='--', linewidth=2)
plt.scatter(optimal_k_math, optimal_score, color='#d7191c', s=100, zorder=5)

plt.annotate(f'Optimal Parsimonious Model\nK={optimal_k_math}, Macro F1={optimal_score:.3f}',
             xy=(optimal_k_math, optimal_score),
             xytext=(optimal_k_math + 2, optimal_score - y_offset),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
             fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#fffacd", ec="gray", lw=1))

plt.title('Multi-label Recursive Feature Elimination (RFECV)\nUnified Mathematical & Clinical Optimization',
          fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Number of Included Features', fontsize=12)
plt.ylabel('Macro F1-Score (Cross-Validated)', fontsize=12)
plt.tight_layout()

plt.savefig('Phase2_Figure_2_OvR_RFECV_Curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Saved 'Phase2_Figure_2_OvR_RFECV_Curve.png'")

# ==============================================================================
# 5. EXPORT FILE FOR THE NEXT STEPS
# ==============================================================================
print(f"\nFinal {optimal_k_math} Selected Features for TCM Multi-label CDSS:")
for feat in selected_features:
    print(f" - {feat}")

columns_to_keep = selected_features + ['Target_AG'] + target_tcm_cols

train_df[columns_to_keep].to_csv('Phase2_Train_Final_K_Features.csv', index=False)
val_df[columns_to_keep].to_csv('Phase2_Val_Final_K_Features.csv', index=False)
test_df[columns_to_keep].to_csv('Phase2_Test_Final_K_Features.csv', index=False)

with open('Phase2_List_of_Final_K_Features.txt', 'w') as f:
    for feat in selected_features:
        f.write(f"{feat}\n")

print("\n--------------------------------------------------")
print("PHASE 2 STEP 2 COMPLETED")
print(f"The pipeline proceeds with a highly robust and compact set of {optimal_k_math} features.")
print("--------------------------------------------------")
