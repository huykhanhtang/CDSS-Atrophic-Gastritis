import pandas as pd, joblib
scaler_1 = joblib.load('CDSS_Feature_Scaler.pkl')
train = pd.read_csv('Train_Final_K_Features.csv')
cols = list(scaler_1.feature_names_in_)

bg_raw    = train[cols].sample(100, random_state=42).reset_index(drop=True)
bg_scaled = pd.DataFrame(scaler_1.transform(bg_raw), columns=cols)
joblib.dump({'raw': bg_raw, 'scaled': bg_scaled}, 'shap_background_phase1.pkl')