import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import log_loss

# ==========================================
# STEP 1: CALCULATE COX-SNELL / NAGELKERKE R² FROM THE TRAINED MODEL
# ==========================================
scaler = joblib.load('CDSS_Feature_Scaler.pkl')
model  = joblib.load('Final_CDSS_Screening_Model.pkl')

train_df = pd.read_csv('Train_Final_K_Features.csv')
y_train  = train_df['Target_AG']
X_train  = scaler.transform(train_df.drop(columns=['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2']))

y_proba  = model.predict_proba(X_train)[:, 1]
ll_model = -log_loss(y_train, y_proba) * len(y_train)

p_bar   = y_train.mean()
ll_null = len(y_train) * (p_bar * np.log(p_bar) + (1 - p_bar) * np.log(1 - p_bar))

n     = len(y_train)
R2_CS = 1 - np.exp((2 / n) * (ll_null - ll_model))

R2_CS_max     = 1 - np.exp(2 * ll_null / n)
R2_Nagelkerke = R2_CS / R2_CS_max

print("=" * 60)
print(f"Train size           : {len(y_train)}")
print(f"Positive cases (AG)  : {y_train.sum()}")
print(f"y_train.mean()       : {y_train.mean():.4f})
print("=" * 60)

print(f"Cox-Snell R² (uncorrected) : {R2_CS:.4f}")
print(f"Nagelkerke R² (corrected)  : {R2_Nagelkerke:.4f}")

# ==========================================
# STEP 2: PMSAMPSIZE FUNCTION (RILEY 2020)
# ==========================================
def pmsampsize_binary(K, R2, prevalence, S_target=0.90, tolerance=0.05, margin_error=0.05):
    import scipy.stats as st

    p = prevalence
    z = st.norm.ppf(0.975)  # 1.96

    S_i = max(20, K) / min(p, 1 - p)

    R2_adj = R2 * 0.637
    S_ii = K / ((1 - S_target) * R2_adj) if (S_target < 1.0 and R2_adj > 0) else np.inf

    S_iii = (z ** 2 * p * (1 - p)) / (margin_error ** 2)

    N_final = int(np.ceil(max(S_i, S_ii, S_iii)))

    return {
        'Criterion_i_EPV':    int(np.ceil(S_i)),
        'Criterion_ii_R2':    int(np.ceil(S_ii)),
        'Criterion_iii_prec': int(np.ceil(S_iii)),
        'Final_N_required':   N_final,
    }


# ==========================================
# APPLICABLE TO RESEARCH
# ==========================================
K = len(scaler.feature_names_in_)

result = pmsampsize_binary(
    K=K,
    R2=R2_Nagelkerke,
    prevalence=y_train.mean(),      
    S_target=0.90,
    tolerance=0.05,
    margin_error=0.05
)
result['Actual_N'] = len(train_df)

print(f"\nFinal variable count of the Phase 1 model (K) = {K}")
print("=== PMSAMPSIZE (RILEY 2020) ===")
for k, v in result.items():
    print(f"  {k:25s}: {v}")
