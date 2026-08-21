import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. DATA LOADING AND FEATURE DEFINITION
# ==============================================================================
print("Loading clinical data and configuring primary/secondary targets...")

# Load dataset
file_path = 'Data.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# Ensure data integrity: Drop records without a Patient ID
if 'Patient_ID' in df.columns:
    df = df.dropna(subset=['Patient_ID'])

# Define targets for the future Clinical Decision Support System (CDSS)
target_1_col = 'Atrophic_Gastritis'
target_2_col_a = 'TCM_Syndromes_1'
target_2_col_b = 'TCM_Syndromes_2'

# Identify columns strictly excluded from the predictive models
excluded_cols = ['Patient_ID', target_1_col, target_2_col_a, target_2_col_b]
endo_cols = [col for col in df.columns if col.startswith('Endo_')]
cols_to_drop = excluded_cols + endo_cols

# Separate Features (X) and Targets (y1, y2, y3)
X_raw = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
y1 = df[target_1_col]
y2 = df[target_2_col_a]
y3 = df[target_2_col_b]

# ==============================================================================
# 2. STRICT DATA SPLITTING (60:20:20) - ZERO DATA LEAKAGE PROTOCOL
# ==============================================================================
print("Splitting cohort into Training (60%), Validation (20%), and Test (20%) sets...")

X_temp, X_test_raw, y1_temp, y1_test, y2_temp, y2_test, y3_temp, y3_test = train_test_split(
    X_raw, y1, y2, y3, test_size=0.20, random_state=42, stratify=y1
)

X_train_raw, X_val_raw, y1_train, y1_val, y2_train, y2_val, y3_train, y3_val = train_test_split(
    X_temp, y1_temp, y2_temp, y3_temp, test_size=0.25, random_state=42, stratify=y1_temp
)

print(f"Data allocation complete -> Train: {len(X_train_raw)} | Validation: {len(X_val_raw)} | Test: {len(X_test_raw)}")

# ==============================================================================
# 3. GENERATE ADVANCED TABLE 1 (STRATIFIED BY AG / NON-AG ACROSS 3 SETS)
# ==============================================================================
print("Generating advanced Table 1 stratified by AG and non-AG with P-values...")


def generate_table_1_stratified(X_train_raw, y1_train, X_val_raw, y1_val, X_test_raw, y1_test):
    train_df = X_train_raw.copy();
    train_df['Target_AG'] = y1_train.values;
    train_df['Cohort'] = 'Train'
    val_df = X_val_raw.copy();
    val_df['Target_AG'] = y1_val.values;
    val_df['Cohort'] = 'Val'
    test_df = X_test_raw.copy();
    test_df['Target_AG'] = y1_test.values;
    test_df['Cohort'] = 'Test'

    df_all = pd.concat([train_df, val_df, test_df], ignore_index=True)

    n_train_non = (train_df['Target_AG'] == 0).sum();
    n_train_ag = (train_df['Target_AG'] == 1).sum()
    n_val_non = (val_df['Target_AG'] == 0).sum();
    n_val_ag = (val_df['Target_AG'] == 1).sum()
    n_test_non = (test_df['Target_AG'] == 0).sum();
    n_test_ag = (test_df['Target_AG'] == 1).sum()

    continuous_vars = ['Dem_Age', 'Dem_Weight', 'Dem_Height', 'Dem_BMI']
    categorical_vars = [col for col in X_train_raw.columns if col not in continuous_vars]

    def format_p(p_val):
        if pd.isna(p_val): return ""
        return "<0.001" if p_val < 0.001 else f"{p_val:.3f}"

    def get_cont_stat(data):
        if len(data) == 0 or data.isna().all(): return "NA"
        return f"{data.mean():.1f} ± {data.std():.1f}"

    def get_cat_stat(data, val):
        count = (data == val).sum()
        percent = (count / len(data)) * 100 if len(data) > 0 else 0
        return f"{count} ({percent:.1f}%)"

    table1_rows = []

    # -------------------------------------------------------------
    # 1. HANDLING CONTINUOUS VARIABLES
    # -------------------------------------------------------------
    for var in continuous_vars:
        if var in df_all.columns:
            row = {
                'Characteristics': var,
                'Subcategory': 'Mean ± SD',
                f'Train non-AG (n={n_train_non})': get_cont_stat(train_df[train_df['Target_AG'] == 0][var]),
                f'Train AG (n={n_train_ag})': get_cont_stat(train_df[train_df['Target_AG'] == 1][var]),
                f'Val non-AG (n={n_val_non})': get_cont_stat(val_df[val_df['Target_AG'] == 0][var]),
                f'Val AG (n={n_val_ag})': get_cont_stat(val_df[val_df['Target_AG'] == 1][var]),
                f'Test non-AG (n={n_test_non})': get_cont_stat(test_df[test_df['Target_AG'] == 0][var]),
                f'Test AG (n={n_test_ag})': get_cont_stat(test_df[test_df['Target_AG'] == 1][var])
            }

            g_non = df_all[df_all['Target_AG'] == 0][var].dropna()
            g_ag = df_all[df_all['Target_AG'] == 1][var].dropna()
            try:
                _, p_ag = stats.ttest_ind(g_non, g_ag, equal_var=False)
            except Exception:
                p_ag = np.nan
            row['P-value (AG vs non-AG)'] = format_p(p_ag)

            try:
                _, p_cohort = stats.f_oneway(train_df[var].dropna(), val_df[var].dropna(), test_df[var].dropna())
            except Exception:
                p_cohort = np.nan
            row['P-value (Cohort Balance)'] = format_p(p_cohort)

            table1_rows.append(row)

    # -------------------------------------------------------------
    # 2. HANDLING CATEGORICAL VARIABLES
    # -------------------------------------------------------------
    for var in categorical_vars:
        if var in df_all.columns:
            unique_vals = sorted(df_all[var].dropna().unique())

            try:
                contingency_ag = pd.crosstab(df_all[var], df_all['Target_AG'])
                _, p_ag, _, _ = stats.chi2_contingency(contingency_ag)
            except Exception:
                p_ag = np.nan

            try:
                contingency_cohort = pd.crosstab(df_all[var], df_all['Cohort'])
                _, p_cohort, _, _ = stats.chi2_contingency(contingency_cohort)
            except Exception:
                p_cohort = np.nan

            for idx, val in enumerate(unique_vals):
                row = {
                    'Characteristics': var if idx == 0 else "",
                    'Subcategory': str(val),
                    f'Train non-AG (n={n_train_non})': get_cat_stat(train_df[train_df['Target_AG'] == 0][var], val),
                    f'Train AG (n={n_train_ag})': get_cat_stat(train_df[train_df['Target_AG'] == 1][var], val),
                    f'Val non-AG (n={n_val_non})': get_cat_stat(val_df[val_df['Target_AG'] == 0][var], val),
                    f'Val AG (n={n_val_ag})': get_cat_stat(val_df[val_df['Target_AG'] == 1][var], val),
                    f'Test non-AG (n={n_test_non})': get_cat_stat(test_df[test_df['Target_AG'] == 0][var], val),
                    f'Test AG (n={n_test_ag})': get_cat_stat(test_df[test_df['Target_AG'] == 1][var], val),
                    'P-value (AG vs non-AG)': format_p(p_ag) if idx == 0 else "",
                    'P-value (Cohort Balance)': format_p(p_cohort) if idx == 0 else ""
                }
                table1_rows.append(row)

    return pd.DataFrame(table1_rows)


table1_stratified_df = generate_table_1_stratified(
    X_train_raw, y1_train,
    X_val_raw, y1_val,
    X_test_raw, y1_test
)
table1_stratified_df.to_csv('Table_1_Baseline_Characteristics_Stratified.csv', index=False)
print("-> Successfully exported 'Table_1_Baseline_Characteristics_Stratified.csv'")

# ==============================================================================
# 4. PREPROCESSING PIPELINE (AVOIDING DATA LEAKAGE)
# ==============================================================================
print("Constructing preprocessing pipeline and encoding targets...")

# --- PREPROCESS INDEPENDENT FEATURES (X) ---
# Automatically detect categorical columns (text/object data type)
categorical_cols = X_train_raw.select_dtypes(include=['object']).columns.tolist()

# The remaining columns are treated as numerical (int/float)
numerical_cols = X_train_raw.select_dtypes(exclude=['object']).columns.tolist()

print(f"Detected {len(categorical_cols)} categorical and {len(numerical_cols)} numerical features.")

# Define transformation pipeline for Categorical features (Impute -> One-Hot Encode)
cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Define transformation pipeline for Numerical features (Impute -> Standardize)
num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Combine pipelines into a unified preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

# CRITICAL: Fit the preprocessor ONLY on the Training set to prevent data leakage
preprocessor.fit(X_train_raw)

# Extract new feature names generated by One-Hot Encoding
ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(
    categorical_cols)
final_feature_names = numerical_cols + list(ohe_feature_names)

# Apply transformation (transform) across all datasets safely
X_train_processed = pd.DataFrame(preprocessor.transform(X_train_raw), columns=final_feature_names)
X_val_processed = pd.DataFrame(preprocessor.transform(X_val_raw), columns=final_feature_names)
X_test_processed = pd.DataFrame(preprocessor.transform(X_test_raw), columns=final_feature_names)

# ==============================================================================
# 5. EXPORT PREPROCESSED DATA FOR SUBSEQUENT RESEARCH STEPS
# ==============================================================================
print("Exporting processed datasets for correlation and modeling phases...")

# Combine features and targets into final DataFrames
train_export = X_train_processed.copy()
train_export['Target_AG'] = y1_train.values
train_export['TCM_Syndromes_1'] = y2_train.values  
train_export['TCM_Syndromes_2'] = y3_train.values  

val_export = X_val_processed.copy()
val_export['Target_AG'] = y1_val.values
val_export['TCM_Syndromes_1'] = y2_val.values
val_export['TCM_Syndromes_2'] = y3_val.values

test_export = X_test_processed.copy()
test_export['Target_AG'] = y1_test.values
test_export['TCM_Syndromes_1'] = y2_test.values
test_export['TCM_Syndromes_2'] = y3_test.values

# Export as CSV files
train_export.to_csv('Processed_Train_Dual_Targets.csv', index=False)
val_export.to_csv('Processed_Val_Dual_Targets.csv', index=False)
test_export.to_csv('Processed_Test_Dual_Targets.csv', index=False)

print("--------------------------------------------------")
print("STEP 1 COMPLETED")
print("Files Generated:")
print("1. Table_1_Baseline_Characteristics.csv")
print("2. Processed_Train_Dual_Targets.csv")
print("3. Processed_Val_Dual_Targets.csv")
print("4. Processed_Test_Dual_Targets.csv")
print("--------------------------------------------------")
