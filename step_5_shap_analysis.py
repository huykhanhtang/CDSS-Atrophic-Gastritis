import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import warnings

# Suppress warnings for cleaner console
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD LASSO-SELECTED DATA (FROM STEP 4)
# ==============================================================================
print("Loading K selected datasets for SHAP analysis...")

train_df = pd.read_csv('CDSS_Web_App/Train_Final_K_Features.csv')

# Isolate features (X) and the primary clinical target (y = Target_AG)
# Drop Target_TCM as we are explaining the Atrophic Gastritis diagnosis model here
y_train = train_df['Target_AG']
X_train = train_df.drop(columns=['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2'])

print(f"Dataset loaded. Ready to explain {X_train.shape[1]} features using SHAP.")

# ==============================================================================
# 2. TRAIN A PRE-TRAINED MODEL (XGBOOST) FOR EXPLANATION
# ==============================================================================
print("Training an XGBoost Classifier to map non-linear clinical relationships...")

# Initialize a standard XGBoost model.
# We use max_depth=4 to prevent overfitting and capture robust general patterns.
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==============================================================================
# 3. COMPUTE SHAP VALUES
# ==============================================================================
print("Calculating Shapley Additive Explanations (SHAP) values...")

# Use TreeExplainer which is highly optimized for XGBoost
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# ==============================================================================
# 4. GENERATE PUBLICATION-READY SHAP PLOTS (FIGURE 4)
# ==============================================================================
print("Generating high-resolution SHAP visualizations (300 DPI)...")


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


# TẠO BẢN SAO DATAFRAME ĐỂ VẼ BIỂU ĐỒ (Bảo vệ dữ liệu gốc)
X_train_display = X_train.copy()
# Áp dụng đổi tên cho tất cả các cột
X_train_display.columns = [format_clinical_name(col) for col in X_train_display.columns]

# --- PLOT 4A: SHAP Summary Plot (Bubble/Swarm Plot) ---
plt.figure(figsize=(12, 10))

# Chú ý: Truyền X_train_display vào hàm vẽ thay vì X_train
shap.summary_plot(
    shap_values,
    X_train_display,
    max_display=35,
    show=False,
    cmap=plt.get_cmap("coolwarm")
)

plt.tight_layout()
plt.savefig('Figure_4A_SHAP_Summary_Bubble_Plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Saved 'Figure_4A_SHAP_Summary_Bubble_Plot.png'")

# --- PLOT 4B: SHAP Bar Plot (Absolute Mean Importance) ---
plt.figure(figsize=(10, 8))

# Chú ý: Truyền X_train_display vào hàm vẽ thay vì X_train
shap.summary_plot(
    shap_values,
    X_train_display,
    plot_type="bar",
    max_display=35,
    show=False,
    color="#4C72B0"
)

plt.xlabel('Mean absolute SHAP value (Average impact on model output magnitude)')
plt.tight_layout()
plt.savefig('Figure_4B_SHAP_Bar_Plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Saved 'Figure_4B_SHAP_Bar_Plot.png'")

# ==============================================================================
# 5. EXPORT SHAP IMPORTANCE SCORES TO CSV
# ==============================================================================
print("Exporting SHAP importance scores for detailed analysis...")

# Calculate the mean absolute SHAP value for each feature
mean_shap_values = np.abs(shap_values).mean(axis=0)

# Create a DataFrame for reporting
shap_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Mean_Abs_SHAP': mean_shap_values
}).sort_values(by='Mean_Abs_SHAP', ascending=False)

shap_df.to_csv('SHAP_Feature_Importance_Scores.csv', index=False)
print("-> Saved 'SHAP_Feature_Importance_Scores.csv'")

print("\n--------------------------------------------------")
print("✅ STEP 5 COMPLETED SUCCESSFULLY.")
print("Files Generated:")
print("1. Figure_4A_SHAP_Summary_Bubble_Plot.png")
print("2. Figure_4B_SHAP_Bar_Plot.png")
print("3. SHAP_Feature_Importance_Scores.csv")
print("--------------------------------------------------")

import joblib

joblib.dump(model, 'SHAP_Explainer_Phase1.pkl')
print("-> Đã xuất mô hình giải thích SHAP_Explainer_Phase1.pkl")