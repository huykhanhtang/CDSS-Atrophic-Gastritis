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

# Tự động nhận diện 6 cột Target TCM nhờ tiền tố "Target_TCM_"
target_tcm_cols = [col for col in train_df.columns if col.startswith('Target_TCM_')]
y_train_tcm = train_df[target_tcm_cols]

# Tách biến X (Loại bỏ cả Target_AG vì Phase 2 chỉ học từ Triệu chứng lâm sàng)
X_train = train_df.drop(columns=['Target_AG'] + target_tcm_cols)

print(f"Detected {len(target_tcm_cols)} independent TCM Syndromes.")
print(f"Starting OvR LASSO feature selection with {X_train.shape[1]} candidate features...")

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ==============================================================================
# 2. INDEPENDENT ONE-VS-REST (OvR) LASSO REGRESSION
# ==============================================================================
print("\nExecuting 6 Independent L1-Regularized Models (One-Vs-Rest Architecture)...")

# Từ điển lưu trữ hệ số của 6 mô hình
ovr_coefficients = {}

for col in target_tcm_cols:
    syndrome_name = col.replace('Target_TCM_', '').replace('_', ' ')
    print(f" -> Training LASSO for: {syndrome_name}")

    # Huấn luyện LASSO độc lập cho từng chứng với Cross-Validation
    # class_weight='balanced' rất quan trọng để xử lý mất cân bằng của từng chứng đơn lẻ
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

    # Lưu hệ số của chứng hậu hiện tại
    ovr_coefficients[col] = lasso_cv.coef_[0]

# ==============================================================================
# 3. AGGREGATE FEATURE IMPORTANCE ACROSS 6 SYNDROMES
# ==============================================================================
print("\nAggregating feature importances across all 6 models...")

# Tạo Dataframe tổng hợp hệ số từ 6 mô hình
coef_df = pd.DataFrame(ovr_coefficients, index=X_train.columns)

# Tính "Trung bình trị tuyệt đối" của từng đặc trưng trên cả 6 mô hình
# (Tính trung bình xem một triệu chứng đóng góp bao nhiêu sức mạnh vào TỔNG THỂ 6 chứng)
coef_df['Mean_Abs_Coef'] = coef_df.abs().mean(axis=1)

# NÂNG CẤP: Chỉ giữ lại các biến có sức mạnh trung bình > 0
# (Những biến bị cả 6 mô hình ép về 0 sẽ bị loại bỏ hoàn toàn)
selected_features_df = coef_df[coef_df['Mean_Abs_Coef'] > 0].sort_values(by='Mean_Abs_Coef', ascending=False)
selected_features_list = selected_features_df.index.tolist()

print(f"\n✅ OvR LASSO successfully filtered down to {len(selected_features_list)} pan-syndrome features.")
print("\nTop 15 Most Influential Features (Macro-Averaged):")
print(selected_features_df[['Mean_Abs_Coef']].head(15).reset_index().rename(columns={'index': 'Feature'}).to_string(
    index=False))

# ==============================================================================
# 4. VISUALIZATION: BAR PLOT OF SELECTED FEATURES
# ==============================================================================
print("\nGenerating OvR Feature Importance Bar Plot for Publication...")


# --- HÀM TỰ ĐỘNG ĐỊNH DẠNG LẠI TÊN BIẾN (CHUẨN Y KHOA) ---
def format_clinical_name(col_name):
    # 1. Bỏ hoàn toàn các tiền tố bệnh nền và hút thuốc
    col_name = col_name.replace('Med_Comorbidity_', '')
    col_name = col_name.replace('Med_Smoking_Status_', '')
    col_name = col_name.replace('Med_H_pylori_Status_', 'H. pylori_')

    # 2. Xử lý nhóm Lưỡi và Rêu lưỡi có tiền tố kép (Đảo ngược tính từ lên trước danh từ)
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

    # 3. Xử lý các biến Lưỡi/Rêu/Mạch đơn giản khác
    elif col_name.startswith('Tongue_'):
        col_name = col_name.replace('Tongue_', '') + '_tongue'
    elif col_name.startswith('Coating_'):
        col_name = col_name.replace('Coating_', '') + '_coating'
    elif col_name.startswith('Pulse_'):
        col_name = col_name.replace('Pulse_', '') + '_pulse'

    # 4. Xóa các tiền tố phân nhóm chung còn lại
    for prefix in ['Sym_', 'Dem_', 'Med_']:
        if col_name.startswith(prefix):
            col_name = col_name.replace(prefix, '')

    # 5. Xóa dấu gạch dưới và tạo khoảng trắng
    col_name = col_name.replace('_', ' ').strip()

    # 6. Viết hoa chữ cái đầu tiên (Sentence case)
    col_name = col_name.capitalize()

    # 7. Sửa lại các từ viết tắt y khoa bị hàm capitalize() ép thành chữ thường
    col_name = col_name.replace('H. pylori', 'H. pylori')
    col_name = col_name.replace('Bmi', 'BMI')

    return col_name


# Trích xuất dữ liệu để vẽ (Top 25 biến)
plot_data = selected_features_df.head(25).reset_index()

# Ứng dụng hàm format_clinical_name để tạo một cột nhãn mới CHỈ DÙNG ĐỂ VẼ (Không ảnh hưởng đến dataframe gốc)
plot_data['Formatted_Feature'] = plot_data['index'].apply(format_clinical_name)

plt.figure(figsize=(12, 10))
sns.barplot(
    data=plot_data,
    x='Mean_Abs_Coef',
    y='Formatted_Feature',  # Dùng cột nhãn mới này làm trục Y
    palette='viridis'
)

plt.xlabel('Mean Absolute Coefficient (Across 6 Independent Syndromes)', fontsize=12)
plt.ylabel('Clinical Features', fontsize=12)
plt.tight_layout()

# Lưu đa định dạng theo chuẩn Tạp chí (SVG, TIFF, PNG)
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
print("✅ PHASE 2 - STEP 1 COMPLETED SUCCESSFULLY.")
print("Files Generated:")
print("1. Phase2_Figure_1_OvR_LASSO_Features.png")
print("2. Phase2_Train_LASSO_Selected.csv")
print("3. Phase2_Val_LASSO_Selected.csv")
print("4. Phase2_Test_LASSO_Selected.csv")
print("--------------------------------------------------")