import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD MULTI-LABEL DATA (FROM PHASE 2 - STEP 0)
# ==============================================================================
print("Loading Multi-Label datasets...")

train_df = pd.read_csv('Phase2_Train_MultiLabel.csv')
val_df = pd.read_csv('Phase2_Val_MultiLabel.csv')
test_df = pd.read_csv('Phase2_Test_MultiLabel.csv')

target_tcm_cols = [col for col in train_df.columns if col.startswith('Target_TCM_')]
y_train_tcm = train_df[target_tcm_cols]

X_train = train_df.drop(columns=['Target_AG'] + target_tcm_cols)

print(f"Detected {len(target_tcm_cols)} independent TCM Syndromes.")
print(f"Starting OvR LASSO feature selection with {X_train.shape[1]} candidate features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ==============================================================================
# 2. INDEPENDENT ONE-VS-REST (OvR) LASSO REGRESSION
# ==============================================================================
print("\nExecuting 6 Independent L1-Regularized Models (One-Vs-Rest Architecture)...")

ovr_coefficients = {}

for col in target_tcm_cols:
    syndrome_name = col.replace('Target_TCM_', '').replace('_', ' ')
    print(f" -> Training LASSO for: {syndrome_name}")

    lasso_cv = LogisticRegressionCV(
        cv=5,
        penalty='l1',
        solver='liblinear',
        class_weight='balanced',
        scoring='neg_log_loss',
        random_state=42,
        n_jobs=-1
    )
    lasso_cv.fit(X_train_scaled, y_train_tcm[col])

    ovr_coefficients[col] = lasso_cv.coef_[0]

# ==============================================================================
# 3. AGGREGATE FEATURE IMPORTANCE ACROSS 6 SYNDROMES
# ==============================================================================
print("\nAggregating feature importances across all 6 models...")

coef_df = pd.DataFrame(ovr_coefficients, index=X_train.columns)

coef_df['Mean_Abs_Coef'] = coef_df.abs().mean(axis=1)

selected_features_df = coef_df[coef_df['Mean_Abs_Coef'] > 0].sort_values(by='Mean_Abs_Coef', ascending=False)
selected_features_list = selected_features_df.index.tolist()

print(f"\n OvR LASSO successfully filtered down to {len(selected_features_list)} pan-syndrome features.")
print("\nTop 15 Most Influential Features (Macro-Averaged):")
print(selected_features_df[['Mean_Abs_Coef']].head(15).reset_index().rename(columns={'index': 'Feature'}).to_string(
    index=False))

# ==============================================================================
# 4. VISUALIZATION: BAR PLOT OF SELECTED FEATURES
# ==============================================================================
print("\nGenerating OvR Feature Importance Bar Plot for Publication...")

def format_clinical_name(col_name):
    col_name = col_name.replace('Med_Comorbidity_', '')
    col_name = col_name.replace('Med_Smoking_Status_', '')
    col_name = col_name.replace('Med_H_pylori_Status_', 'H. pylori_')

    if col_name.startswith('Tongue_Movement_'):
        col_name = col_name.replace('Tongue_Movement_', '') + '_tongue'
    elif col_name.startswith('Tongue_Color_'):
        col_name = col_name.replace('Tongue_Color_', '') + '_tongue'
    elif col_name.startswith('Tongue_Shape_'):
        col_name = col_name.replace('Tongue_Shape_', '') + '_tongue_shape'
    elif col_name.startswith('Tongue_Moisture_'):
        col_name = col_name.replace('Tongue_Moisture_', '') + '_tongue'
    elif col_name.startswith('Coating_Thickness_'):
        col_name = col_name.replace('Coating_Thickness_', '') + '_coating'
    elif col_name.startswith('Coating_Color_'):
        col_name = col_name.replace('Coating_Color_', '') + '_coating'

    elif col_name.startswith('Tongue_'):
        col_name = col_name.replace('Tongue_', '') + '_tongue'
    elif col_name.startswith('Coating_'):
        col_name = col_name.replace('Coating_', '') + '_coating'
    elif col_name.startswith('Pulse_'):
        col_name = col_name.replace('Pulse_', '') + '_pulse'

    for prefix in ['Sym_', 'Dem_', 'Med_']:
        if col_name.startswith(prefix):
            col_name = col_name.replace(prefix, '')

    col_name = col_name.replace('_', ' ').strip()

    col_name = col_name.capitalize()

    col_name = col_name.replace('H. pylori', 'H. pylori')
    col_name = col_name.replace('Bmi', 'BMI')

    return col_name

plot_data = selected_features_df.head(25).reset_index()

plot_data['Formatted_Feature'] = plot_data['index'].apply(format_clinical_name)

plt.figure(figsize=(12, 10))
sns.barplot(
    data=plot_data,
    x='Mean_Abs_Coef',
    y='Formatted_Feature',  
    palette='viridis'
)

plt.xlabel('Mean Absolute Coefficient (Across 6 Independent Syndromes)', fontsize=12)
plt.ylabel('Clinical Features', fontsize=12)
plt.tight_layout()

plt.savefig('Supplemental_Figure_S4_OvR_LASSO_Features.svg', format="svg", bbox_inches='tight')
plt.savefig('Supplemental_Figure_S4_OvR_LASSO_Features.tiff', format="tiff", dpi=600, bbox_inches='tight',
            pil_kwargs={"compression": "tiff_lzw"})

plt.close()
print("-> Saved 'Supplemental_Figure_S4_OvR_LASSO_Features.svg' and '.tiff'")

# ==============================================================================
# 5. EXPORT REDUCED DATASETS FOR NEXT STEP
# ==============================================================================
print("\nExporting pristine Phase 2 LASSO datasets...")

columns_to_keep = selected_features_list + ['Target_AG'] + target_tcm_cols

train_df[columns_to_keep].to_csv('Phase2_Train_LASSO_Selected.csv', index=False)
val_df[columns_to_keep].to_csv('Phase2_Val_LASSO_Selected.csv', index=False)
test_df[columns_to_keep].to_csv('Phase2_Test_LASSO_Selected.csv', index=False)

print("--------------------------------------------------")
print("PHASE 2 - STEP 1 COMPLETED")
print("Files Generated:")
print("1. Phase2_Figure_1_OvR_LASSO_Features.png")
print("2. Phase2_Train_LASSO_Selected.csv")
print("3. Phase2_Val_LASSO_Selected.csv")
print("4. Phase2_Test_LASSO_Selected.csv")
print("--------------------------------------------------")
