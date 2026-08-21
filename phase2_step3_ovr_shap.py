import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import warnings

# Suppress warnings for cleaner console
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD MULTI-LABEL PARSIMONIOUS DATASETS (K=34)
# ==============================================================================
print("Loading K-selected datasets for Multi-label SHAP analysis...")

train_df = pd.read_csv('Phase2_Train_Final_K_Features.csv')

# Nhận diện 6 cột Target
target_tcm_cols = [col for col in train_df.columns if col.startswith('Target_TCM_')]

# Isolate features (X)
# Chú ý: Bỏ Target_AG và 6 Target_TCM để AI chỉ học thuần túy từ 34 triệu chứng
X_train = train_df.drop(columns=['Target_AG'] + target_tcm_cols)

print(f"Dataset loaded successfully. {X_train.shape[1]} features ready for Multi-label SHAP Explanation.")


# --- HÀM TỰ ĐỘNG ĐỊNH DẠNG LẠI TÊN BIẾN (CHUẨN Y KHOA) ---
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


# TẠO BẢN SAO DATAFRAME ĐỂ VẼ BIỂU ĐỒ & XUẤT CSV (Bảo vệ dữ liệu gốc)
X_train_display = X_train.copy()
formatted_columns = [format_clinical_name(col) for col in X_train_display.columns]
X_train_display.columns = formatted_columns

# ==============================================================================
# 2. ITERATIVE SHAP ANALYSIS FOR 6 INDEPENDENT SYNDROMES
# ==============================================================================
print("\nInitiating XGBoost Explainer for 6 independent TCM Syndromes...")

# Tạo DataFrame lưu trữ tầm quan trọng, sử dụng danh sách tên biến đã format đẹp
shap_importance_matrix = pd.DataFrame(index=formatted_columns)

for col in target_tcm_cols:
    # Xử lý tên chứng hậu cho đẹp để in lên biểu đồ
    syndrome_raw_name = col.replace('Target_TCM_', '')
    syndrome_display_name = syndrome_raw_name.replace('_', ' ')

    print(f"\n--- Analyzing Mechanism for: {syndrome_display_name} ---")

    y_train = train_df[col]

    # 2.1 Huấn luyện mô hình XGBoost nhị phân (Binary) cho riêng chứng hậu này
    # Cài đặt scale_pos_weight có thể giúp cân bằng class nếu dữ liệu hiếm
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    # Lưu ý: Model vẫn học trên X_train (dữ liệu gốc) để đảm bảo tính chính xác
    model.fit(X_train, y_train)

    # 2.2 Tính toán giá trị SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Lưu giá trị SHAP trung bình tuyệt đối vào ma trận tổng
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance_matrix[syndrome_raw_name] = mean_abs_shap

    # 2.3 Vẽ biểu đồ SHAP Summary Plot (Bubble Plot)
    plt.figure(figsize=(10, 8))

    # TRUYỀN X_train_display VÀO HÀM VẼ ĐỂ HIỂN THỊ NHÃN ĐẸP
    shap.summary_plot(
        shap_values,
        X_train_display,
        max_display=12,  # Chỉ hiện Top 12 biến định đoạt mạnh nhất cho chứng này
        show=False,
        cmap=plt.get_cmap("coolwarm")
    )

    plt.title(f'Diagnostic Mechanism for {syndrome_display_name}',
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    # Lưu ảnh với tên riêng biệt
    filename = f'Phase2_Figure_3_SHAP_{syndrome_raw_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f" -> Saved diagnostic plot: {filename}")

# ==============================================================================
# 3. EXPORT MASTER SHAP IMPORTANCE MATRIX
# ==============================================================================
print("\nExporting Master SHAP Importance Matrix to CSV...")

# Thêm cột tổng sức mạnh (Sum of SHAP values across 6 syndromes) để xếp hạng
shap_importance_matrix['Total_Impact'] = shap_importance_matrix.sum(axis=1)
shap_importance_matrix = shap_importance_matrix.sort_values(by='Total_Impact', ascending=False)

# Đẩy index (tên biến đã format) thành một cột
shap_importance_matrix.reset_index(inplace=True)
shap_importance_matrix.rename(columns={'index': 'Feature'}, inplace=True)

shap_importance_matrix.to_csv('Phase2_Table_S1_SHAP_Matrix.csv', index=False)
print("-> Saved 'Phase2_Table_S1_SHAP_Matrix.csv'")

print("\n--------------------------------------------------")
print("✅ PHASE 2 STEP 3 COMPLETED SUCCESSFULLY.")
print("Bạn đã có 6 biểu đồ lột tả cơ chế chẩn đoán riêng biệt cho 6 chứng hậu!")
print("Đồng thời file CSV cũng đã được format tên biến chuẩn Y khoa.")
print("--------------------------------------------------")