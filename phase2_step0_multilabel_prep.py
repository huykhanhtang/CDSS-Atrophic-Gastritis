import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD PROCESSED DATASETS (FROM PHASE 1 - STEP 1)
# ==============================================================================
print("PHASE 2 - STEP 0: MULTI-LABEL DATA TRANSFORMATION")
print("Loading preprocessed datasets from Phase 1...")

try:
    train_df = pd.read_csv('Processed_Train_Dual_Targets.csv')
    val_df = pd.read_csv('Processed_Val_Dual_Targets.csv')
    test_df = pd.read_csv('Processed_Test_Dual_Targets.csv')
    print("Successfully loaded Train, Validation, and Test datasets.")
except FileNotFoundError:
    print("ERROR: Files 'Processed_..._Dual_Targets.csv' not found. Please check Step 1 of Phase 1.")
    exit()

# ==============================================================================
# 2. COMBINE DUAL SYNDROME COLUMNS INTO A LIST
# ==============================================================================
print("\nConsolidating TCM Syndromes into a unified format...")

def extract_syndrome_list(row):
    
    syndromes = []
    if pd.notna(row['TCM_Syndromes_1']) and str(row['TCM_Syndromes_1']).strip() != '':
        syndromes.append(str(row['TCM_Syndromes_1']).strip())
    if pd.notna(row['TCM_Syndromes_2']) and str(row['TCM_Syndromes_2']).strip() != '':
        syndromes.append(str(row['TCM_Syndromes_2']).strip())

    return list(set(syndromes))

train_df['Syndrome_List'] = train_df.apply(extract_syndrome_list, axis=1)
val_df['Syndrome_List'] = val_df.apply(extract_syndrome_list, axis=1)
test_df['Syndrome_List'] = test_df.apply(extract_syndrome_list, axis=1)

# ==============================================================================
# 3. APPLY MULTI-LABEL BINARIZER (THE OVR FOUNDATION)
# ==============================================================================
print("\nApplying MultiLabelBinarizer to create independent targets...")

mlb = MultiLabelBinarizer()

mlb.fit(train_df['Syndrome_List'])
syndrome_classes = mlb.classes_

print(f"Detected {len(syndrome_classes)} unique TCM Syndromes:")
for cls in syndrome_classes:
    print(f" - {cls}")

def transform_and_format(df, dataset_name):
    
    encoded_matrix = mlb.transform(df['Syndrome_List'])

    target_columns = [f"Target_TCM_{cls}" for cls in syndrome_classes]
    encoded_df = pd.DataFrame(encoded_matrix, columns=target_columns, index=df.index)

    cols_to_drop = ['TCM_Syndromes_1', 'TCM_Syndromes_2', 'Syndrome_List']
    final_df = pd.concat([df.drop(columns=cols_to_drop), encoded_df], axis=1)

    print(f"{dataset_name} transformed successfully. New shape: {final_df.shape}")
    return final_df, target_columns

train_ml, generated_targets = transform_and_format(train_df, "Training Set")
val_ml, _ = transform_and_format(val_df, "Validation Set")
test_ml, _ = transform_and_format(test_df, "Test Set")

# ==============================================================================
# 4. EXPORT READY-TO-USE MULTI-LABEL DATASETS
# ==============================================================================
print("\nExporting datasets for Phase 2 - Step 1 (TCM LASSO)...")

train_ml.to_csv('Phase2_Train_MultiLabel.csv', index=False)
val_ml.to_csv('Phase2_Val_MultiLabel.csv', index=False)
test_ml.to_csv('Phase2_Test_MultiLabel.csv', index=False)

with open('CDSS_Web_App/Phase2_Target_Columns.txt', 'w') as f:
    for target in generated_targets:
        f.write(f"{target}\n")

print("--------------------------------------------------")
print("PHASE 2 - STEP 0 COMPLETED")
print("Files Generated:")
print("1. Phase2_Train_MultiLabel.csv")
print("2. Phase2_Val_MultiLabel.csv")
print("3. Phase2_Test_MultiLabel.csv")
print("4. Phase2_Target_Columns.txt")
print("--------------------------------------------------")
