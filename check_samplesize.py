import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import log_loss

# ==========================================
# BƯỚC 1: TÍNH R² COX-SNELL / NAGELKERKE TỪ MÔ HÌNH ĐÃ TRAIN
# ==========================================
scaler = joblib.load('CDSS_Feature_Scaler.pkl')
model  = joblib.load('Final_CDSS_Screening_Model.pkl')

train_df = pd.read_csv('Train_Final_K_Features.csv')
y_train  = train_df['Target_AG']
X_train  = scaler.transform(train_df.drop(columns=['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2']))

# 1. Log-likelihood của mô hình
y_proba  = model.predict_proba(X_train)[:, 1]
ll_model = -log_loss(y_train, y_proba) * len(y_train)

# 2. Log-likelihood của mô hình null (chỉ intercept)
p_bar   = y_train.mean()
ll_null = len(y_train) * (p_bar * np.log(p_bar) + (1 - p_bar) * np.log(1 - p_bar))

# 3. Cox-Snell R² (uncorrected)
n     = len(y_train)
R2_CS = 1 - np.exp((2 / n) * (ll_null - ll_model))

# 4. Nagelkerke R² (corrected)
R2_CS_max     = 1 - np.exp(2 * ll_null / n)
R2_Nagelkerke = R2_CS / R2_CS_max

# ✅ IN RA PREVALENCE THẬT CỦA TẬP TRAIN
print("=" * 60)
print(f"Train size           : {len(y_train)}")
print(f"Positive cases (AG)  : {y_train.sum()}")
print(f"y_train.mean()       : {y_train.mean():.4f}  ← prevalence dùng cho pmsampsize")
print("=" * 60)

print(f"Cox-Snell R² (uncorrected) : {R2_CS:.4f}")
print(f"Nagelkerke R² (corrected)  : {R2_Nagelkerke:.4f}")


# ==========================================
# BƯỚC 2: HÀM PMSAMPSIZE (RILEY 2020)
# ==========================================
def pmsampsize_binary(K, R2, prevalence, S_target=0.90, tolerance=0.05, margin_error=0.05):
    """
    Tính cỡ mẫu tối thiểu theo khung Riley et al. (BMJ 2020, m441)
    cho bài toán phân loại nhị phân.
    """
    import scipy.stats as st

    p = prevalence
    z = st.norm.ppf(0.975)  # 1.96

    # CRITERION (i): EPV tối thiểu
    S_i = max(20, K) / min(p, 1 - p)

    # CRITERION (ii): Shrinkage ≥ S_target & optimism R² ≤ tolerance
    R2_adj = R2 * 0.637
    S_ii = K / ((1 - S_target) * R2_adj) if (S_target < 1.0 and R2_adj > 0) else np.inf

    # CRITERION (iii): Precision của prevalence
    S_iii = (z ** 2 * p * (1 - p)) / (margin_error ** 2)

    N_final = int(np.ceil(max(S_i, S_ii, S_iii)))

    return {
        'Criterion_i_EPV':    int(np.ceil(S_i)),
        'Criterion_ii_R2':    int(np.ceil(S_ii)),
        'Criterion_iii_prec': int(np.ceil(S_iii)),
        'Final_N_required':   N_final,
    }


# ==========================================
# ÁP DỤNG CHO NGHIÊN CỨU
# ==========================================
K = len(scaler.feature_names_in_)   # Tự đọc số biến cuối từ scaler

result = pmsampsize_binary(
    K=K,
    R2=R2_Nagelkerke,
    prevalence=y_train.mean(),      # ✅ Dùng prevalence thật từ train set
    S_target=0.90,
    tolerance=0.05,
    margin_error=0.05
)
result['Actual_N'] = len(train_df)  # N thật của tập train

print(f"\nSố biến cuối của mô hình Phase 1 (K) = {K}")
print("=== PMSAMPSIZE (RILEY 2020) ===")
for k, v in result.items():
    print(f"  {k:25s}: {v}")