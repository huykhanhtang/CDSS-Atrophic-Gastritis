import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD LASSO DATASETS
# ==============================================================================
print("Loading LASSO-selected datasets for Phase 1 Analysis...")
train_df = pd.read_csv('Train_LASSO_Selected.csv')
val_df = pd.read_csv('Val_LASSO_Selected.csv')
test_df = pd.read_csv('Test_LASSO_Selected.csv')

y_train = train_df['Target_AG']
X_train = train_df.drop(columns=['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2'])

available_features = X_train.shape[1]
print(f"Candidate features available from LASSO: {available_features}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ==============================================================================
# 2. PURE MATHEMATICAL OPTIMIZATION (RFECV Log-Loss)
# ==============================================================================
estimator = LogisticRegression(C=1, solver='liblinear', class_weight='balanced', random_state=42, max_iter=2000)

print("\nExecuting Binary RFECV using Negative Log-Loss...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rfecv = RFECV(
    estimator=estimator,
    step=1,
    cv=cv,
    scoring='neg_log_loss',
    min_features_to_select=3, 
    n_jobs=-1
)

rfecv.fit(X_train_scaled, y_train)

optimal_k_math = rfecv.n_features_
print(f"RFECV Optimization successfully converged at K = {optimal_k_math} features.")

# ==============================================================================
# 3. EXTRACT VARIABLE LIST
# ==============================================================================

selected_features = X_train.columns[rfecv.support_].tolist()

# ==============================================================================
# 4. VISUALIZATION (FIGURE 3)
# ==============================================================================
print("Generating Phase 1 RFECV Curve (Figure 3)...")
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

cv_results = rfecv.cv_results_['mean_test_score']
min_features = rfecv.min_features_to_select
x_axis = range(min_features, min_features + len(cv_results))

plt.plot(x_axis, cv_results, marker='o', linestyle='-', color='#2c7bb6', linewidth=2, markersize=5)

y_min = min(cv_results)
y_max = max(cv_results)
y_range = y_max - y_min
if y_range < 0.05:
    plt.ylim(y_max - 0.02, min(0, y_max + 0.005))
else:
    margin = y_range * 0.05
    plt.ylim(y_min - margin, min(0, y_max + margin))
y_offset = (plt.ylim()[1] - plt.ylim()[0]) * 0.15

optimal_idx = optimal_k_math - min_features
optimal_score = cv_results[optimal_idx]

plt.axvline(x=optimal_k_math, color='#d7191c', linestyle='--', linewidth=2)
plt.scatter(optimal_k_math, optimal_score, color='#d7191c', s=100, zorder=5)

plt.annotate(f'Optimal Parsimonious Model\nK={optimal_k_math}, Neg Log-Loss={optimal_score:.3f}',
             xy=(optimal_k_math, optimal_score),
             xytext=(optimal_k_math - 1.5, optimal_score - y_offset),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
             fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#fffacd", ec="gray", lw=1))

plt.xlabel('Number of Included Features', fontsize=12)
plt.ylabel('Negative Log-Loss (Cross-Validated)', fontsize=12)
plt.tight_layout()
plt.savefig('Figure_3_RFECV_Curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Saved 'Figure_3_RFECV_Curve.png'")

# ==============================================================================
# 5. EXPORT FILE FOR THE NEXT STEPS
# ==============================================================================
print(f"\nFinal {optimal_k_math} Selected Features for AG Screening CDSS:")
for feat in selected_features:
    print(f" - {feat}")

columns_to_keep = selected_features + ['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2']

train_df[columns_to_keep].to_csv('Train_Final_K_Features.csv', index=False)
val_df[columns_to_keep].to_csv('Val_Final_K_Features.csv', index=False)
test_df[columns_to_keep].to_csv('Test_Final_K_Features.csv', index=False)

with open('List_of_Final_K_Features.txt', 'w') as f:
    for feat in selected_features:
        f.write(f"{feat}\n")

print("\n--------------------------------------------------")
print("STEP 4 COMPLETED")
print(f"The pipeline proceeds with a highly robust and compact set of {optimal_k_math} features.")
print("--------------------------------------------------")
