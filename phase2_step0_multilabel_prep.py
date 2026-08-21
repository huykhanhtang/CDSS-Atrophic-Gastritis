import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

# Suppress warnings for cleaner console output
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD PROCESSED DATASETS (FROM PHASE 1 - STEP 1)
# ==============================================================================
print("PHASE 2 - STEP 0: MULTI-LABEL DATA TRANSFORMATION")
print("Loading preprocessed datasets from Phase 1...")

# Đọc các file đã được Impute và Scale từ cuối Phase 1
try:
    train_df = pd.read_csv('Processed_Train_Dual_Targets.csv')
    val_df = pd.read_csv('Processed_Val_Dual_Targets.csv')
    test_df = pd.read_csv('Processed_Test_Dual_Targets.csv')
    print("✅ Successfully loaded Train, Validation, and Test datasets.")
except FileNotFoundError:
    print("❌ LỖI: Không tìm thấy các file 'Processed_..._Dual_Targets.csv'. Vui lòng kiểm tra lại Step 1 của Phase 1.")
    exit()

# ==============================================================================
# 2. COMBINE DUAL SYNDROME COLUMNS INTO A LIST
# ==============================================================================
print("\nConsolidating TCM Syndromes into a unified format...")


def extract_syndrome_list(row):
    """
    Hàm nội bộ: Trích xuất các chứng hậu từ 2 cột và gom thành 1 danh sách.
    Loại bỏ các giá trị trống (NaN) nếu bệnh nhân chỉ có 1 chứng hậu.
    """
    syndromes = []
    # Kiểm tra cột 1
    if pd.notna(row['TCM_Syndromes_1']) and str(row['TCM_Syndromes_1']).strip() != '':
        syndromes.append(str(row['TCM_Syndromes_1']).strip())
    # Kiểm tra cột 2
    if pd.notna(row['TCM_Syndromes_2']) and str(row['TCM_Syndromes_2']).strip() != '':
        syndromes.append(str(row['TCM_Syndromes_2']).strip())

    # Loại bỏ các giá trị trùng lặp (nếu vô tình nhập trùng ở cả 2 cột)
    return list(set(syndromes))


# Áp dụng hàm cho cả 3 tập dữ liệu
train_df['Syndrome_List'] = train_df.apply(extract_syndrome_list, axis=1)
val_df['Syndrome_List'] = val_df.apply(extract_syndrome_list, axis=1)
test_df['Syndrome_List'] = test_df.apply(extract_syndrome_list, axis=1)

# ==============================================================================
# 3. APPLY MULTI-LABEL BINARIZER (THE OVR FOUNDATION)
# ==============================================================================
print("\nApplying MultiLabelBinarizer to create independent targets...")

# Khởi tạo thuật toán Binarizer
mlb = MultiLabelBinarizer()

# CHỈ 'fit' trên tập Train để chuẩn hóa danh sách các chứng hậu
mlb.fit(train_df['Syndrome_List'])
syndrome_classes = mlb.classes_

print(f"Detected {len(syndrome_classes)} unique TCM Syndromes:")
for cls in syndrome_classes:
    print(f" - {cls}")


def transform_and_format(df, dataset_name):
    """
    Biến đổi danh sách chứng hậu thành ma trận 0/1 và ghép trở lại dataframe gốc.
    """
    # Tạo ma trận nhị phân (0 và 1)
    encoded_matrix = mlb.transform(df['Syndrome_List'])

    # Đặt tên cột theo đúng chuẩn quy ước: Target_TCM_TênChứngHậu
    target_columns = [f"Target_TCM_{cls}" for cls in syndrome_classes]
    encoded_df = pd.DataFrame(encoded_matrix, columns=target_columns, index=df.index)

    # Ghép ma trận này vào bộ dữ liệu gốc, đồng thời xóa bỏ 2 cột chữ cũ
    cols_to_drop = ['TCM_Syndromes_1', 'TCM_Syndromes_2', 'Syndrome_List']
    final_df = pd.concat([df.drop(columns=cols_to_drop), encoded_df], axis=1)

    print(f"{dataset_name} transformed successfully. New shape: {final_df.shape}")
    return final_df, target_columns


# Thực thi biến đổi
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

# Lưu danh sách 6 cột target ra một file txt để các bước sau tự động đọc,
# tránh việc bạn phải gõ lại tên 6 cột này bằng tay trong code.
with open('CDSS_Web_App/Phase2_Target_Columns.txt', 'w') as f:
    for target in generated_targets:
        f.write(f"{target}\n")

print("--------------------------------------------------")
print("✅ PHASE 2 - STEP 0 COMPLETED SUCCESSFULLY.")
print("Files Generated:")
print("1. Phase2_Train_MultiLabel.csv")
print("2. Phase2_Val_MultiLabel.csv")
print("3. Phase2_Test_MultiLabel.csv")
print("4. Phase2_Target_Columns.txt")
print("--------------------------------------------------")