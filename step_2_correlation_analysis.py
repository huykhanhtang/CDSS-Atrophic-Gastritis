import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings for cleaner console output
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD PREPROCESSED DATA (FROM STEP 1)
# ==============================================================================
print("Loading preprocessed datasets...")

# Load the datasets generated in Step 1
train_df = pd.read_csv('Processed_Train_Dual_Targets.csv')
val_df = pd.read_csv('Processed_Val_Dual_Targets.csv')
test_df = pd.read_csv('Processed_Test_Dual_Targets.csv')

# --- FIX LỖI Ở ĐÂY ---
# Isolate features (X) from targets (y) on the Training set to calculate correlations
# Cập nhật thành 2 cột TCM mới xuất ra từ Step 1
target_cols = ['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2']
X_train = train_df.drop(columns=target_cols)

print(f"Initial feature count for correlation analysis: {X_train.shape[1]}")

# ==============================================================================
# 2. CALCULATE PEARSON CORRELATION MATRIX
# ==============================================================================
print("Calculating Pearson correlation matrix...")
corr_matrix = X_train.corr(method='pearson')

# ==============================================================================
# 3. HIGH-RESOLUTION VISUALIZATION FOR Q1 JOURNAL (HEATMAP)
# ==============================================================================
print("Generating publication-ready Heatmap...")


# --- HÀM TỰ ĐỘNG ĐỊNH DẠNG LẠI TÊN BIẾN (PHIÊN BẢN NÂNG CẤP) ---
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


# Tạo danh sách nhãn đã được format đẹp đẽ
formatted_labels = [format_clinical_name(col) for col in corr_matrix.columns]

# Set the aesthetic style of the plots
sns.set_theme(style="white")

# Generate a mask for the upper triangle to remove duplicate visual information
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Set up the matplotlib figure with a large size to accommodate many features
plt.figure(figsize=(24, 20))

# Generate a custom diverging colormap (Blue for negative, Red for positive)
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# THAY ĐỔI QUAN TRỌNG: Gắn formatted_labels vào trục X và Y
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},
            xticklabels=formatted_labels, yticklabels=formatted_labels)

# Tinh chỉnh tiêu đề và kích cỡ chữ cho đẹp mắt
plt.title('Pearson Correlation Matrix of Clinical Features', fontsize=26, pad=20, fontweight='bold')
plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

# Lưu định dạng vector và TIFF chất lượng cao
plt.savefig('Supplemental_Figure_S1_Correlation_Heatmap.svg', format="svg", bbox_inches='tight')
print("-> Saved 'Supplemental_Figure_S1_Correlation_Heatmap.svg'")

plt.savefig('Supplemental_Figure_S1_Correlation_Heatmap.tiff', format="tiff", dpi=600, bbox_inches='tight',
            pil_kwargs={"compression": "tiff_lzw"})
print("-> Saved 'Supplemental_Figure_S1_Correlation_Heatmap.tiff'")

plt.close()

# ==============================================================================
# 4. IDENTIFY AND REMOVE HIGHLY CORRELATED FEATURES (|r| >= 0.8)
# ==============================================================================
CORRELATION_THRESHOLD = 0.80
print(f"\nScanning for highly correlated feature pairs (|r| >= {CORRELATION_THRESHOLD})...")

# Extract the upper triangle of the correlation matrix (without the diagonal)
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than the threshold
to_drop = []
highly_correlated_pairs = []

for column in upper_tri.columns:
    for row in upper_tri.index:
        correlation_value = upper_tri.loc[row, column]
        if abs(correlation_value) >= CORRELATION_THRESHOLD:
            highly_correlated_pairs.append((row, column, correlation_value))
            # We flag the 'column' feature to be dropped, keeping the 'row' feature
            if column not in to_drop:
                to_drop.append(column)

# Display the findings
if highly_correlated_pairs:
    print("\n⚠️ Highly Correlated Pairs Found:")
    for pair in highly_correlated_pairs:
        print(f"   - {pair[0]} & {pair[1]}: r = {pair[2]:.3f}")
    print(f"\nFeatures flagged for removal ({len(to_drop)} features): {to_drop}")
else:
    print("\n✅ No highly correlated features found. The dataset is optimal.")

# ==============================================================================
# 5. DROP REDUNDANT FEATURES AND EXPORT CLEANED DATASETS
# ==============================================================================
if len(to_drop) > 0:
    print("\nDropping redundant features from Train, Val, and Test sets to prevent Data Leakage...")
    train_clean = train_df.drop(columns=to_drop)
    val_clean = val_df.drop(columns=to_drop)
    test_clean = test_df.drop(columns=to_drop)
else:
    train_clean, val_clean, test_clean = train_df, val_df, test_df

# Save the cleaned datasets
train_clean.to_csv('Train_No_Collinear.csv', index=False)
val_clean.to_csv('Val_No_Collinear.csv', index=False)
test_clean.to_csv('Test_No_Collinear.csv', index=False)

print("\n--------------------------------------------------")
print("✅ STEP 2 COMPLETED SUCCESSFULLY.")
print(f"Final feature count for modeling: {train_clean.shape[1] - 3} (Excluded 3 target columns)")
print("Files Generated:")
print("1. Supplemental_Figure_S1_Correlation_Heatmap.svg")
print("2. Supplemental_Figure_S1_Correlation_Heatmap.tiff")
print("3. Train_No_Collinear.csv")
print("4. Val_No_Collinear.csv")
print("5. Test_No_Collinear.csv")
print("--------------------------------------------------")