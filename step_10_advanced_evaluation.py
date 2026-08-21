import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import scipy.stats as st
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')


# ==============================================================================
# 1. DELONG TEST IMPLEMENTATION
# ==============================================================================
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1D):
    m = predictions_sorted_transposed.shape[1]
    n = predictions_sorted_transposed.shape[0]
    positive_examples = label_1D == 1
    negative_examples = label_1D == 0
    tx = np.empty([m, positive_examples.sum()], dtype=np.float64)
    ty = np.empty([m, negative_examples.sum()], dtype=np.float64)
    tz = np.empty([m, n], dtype=np.float64)
    for r in range(m):
        tz[r, :] = compute_midrank(predictions_sorted_transposed[:, r])
        tx[r, :] = compute_midrank(predictions_sorted_transposed[positive_examples, r])
        ty[r, :] = compute_midrank(predictions_sorted_transposed[negative_examples, r])
    aucs = tz[:, positive_examples].sum(axis=1) / (positive_examples.sum() * negative_examples.sum()) - \
           (positive_examples.sum() + 1.0) / 2.0 / negative_examples.sum()
    v01 = (tz[:, positive_examples] - tx[:, :]) / negative_examples.sum()
    v10 = 1.0 - (tz[:, negative_examples] - ty[:, :]) / positive_examples.sum()
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / positive_examples.sum() + sy / negative_examples.sum()
    return aucs, delongcov


def calc_delong_pvalue(preds1, preds2, label):
    preds = np.array([preds1, preds2]).T
    aucs, delongcov = fastDeLong(preds, label)
    
    var_diff = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    if var_diff == 0:
        return 0, 1.0  # Models are practically identical
    z_stat = (aucs[0] - aucs[1]) / np.sqrt(var_diff)
    p_value = 10 ** (st.norm.logsf(np.abs(z_stat)) / np.log(10)) * 2  # 2-tailed
    return abs(z_stat), p_value


# ==============================================================================
# 2. LOAD DATA AND RECONSTRUCT ALL 9 MODELS
# ==============================================================================
print("Loading datasets, scaler, and reconstructing all 9 models...")

datasets = {
    'Training set': pd.read_csv('CDSS_Web_App/Train_Final_K_Features.csv'),
    'Internal validation set': pd.read_csv('Val_Final_K_Features.csv'),
    'Internal hold-out test set': pd.read_csv('Test_Final_K_Features.csv')
}

scaler = joblib.load('CDSS_Feature_Scaler.pkl')

base_model_names = ['LogisticRegression', 'AdaBoost', 'BernoulliNB', 'KNeighbors', 'XGBoost', 'SVC', 'RandomForest']
models = {}
estimators_for_ensemble = []

for name in base_model_names:
    model = joblib.load(f'TunedModel_{name}.pkl')
    models[name] = model
    # We use a subset of stable models for ensembles to prevent overfitting (as decided in Step 8)
    if name in ['LogisticRegression', 'RandomForest', 'BernoulliNB', 'SVC', 'AdaBoost']:
        estimators_for_ensemble.append((name, model))

print("Fitting Ensemble Models (Soft Voting and Stacking)...")
X_train = scaler.transform(datasets['Training set'].drop(columns=['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2']))
y_train = datasets['Training set']['Target_AG'].values

models['Soft voting'] = VotingClassifier(estimators=estimators_for_ensemble, voting='soft', n_jobs=-1)
models['Stacking'] = StackingClassifier(
    estimators=estimators_for_ensemble,
    final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
    cv=5, n_jobs=-1
)

models['Soft voting'].fit(X_train, y_train)
models['Stacking'].fit(X_train, y_train)

ordered_model_names = base_model_names + ['Soft voting', 'Stacking']

# ==============================================================================
# 3. VISUALIZE ROC CURVES FOR ALL 9 MODELS ACROSS 3 SETS (FIGURE 3)
# ==============================================================================
print("Generating Figure 3: ROC Curves across Train, Val, and Test Sets...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Diagnostic performance of machine learning models for Atrophic Gastritis risk prediction', fontsize=18,
             fontweight='bold', y=1.05)

colors = sns.color_palette("tab10", 9)
color_map = {name: colors[i] for i, name in enumerate(ordered_model_names)}

for idx, (set_name, df) in enumerate(datasets.items()):
    ax = axes[idx]
    y_true = df['Target_AG'].values
    X_scaled = scaler.transform(df.drop(columns=['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2']))

    for model_name in ordered_model_names:
        y_proba = models[model_name].predict_proba(X_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        lw = 2.5 if model_name == 'Stacking' else 1.2
        ls = '-' if model_name in ['Stacking', 'Soft voting'] else '--'
        ax.plot(fpr, tpr, color=color_map[model_name], lw=lw, linestyle=ls, label=f'{model_name} (AUC = {auc:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_title(set_name, fontsize=14, fontweight='bold')
    ax.set_xlabel('1 - Specificity', fontsize=12)
    ax.set_ylabel('Sensitivity', fontsize=12)
    ax.legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig('Figure_3_ROC_Curves_Subplots.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Saved 'Figure_3_ROC_Curves_Subplots.png'")

# ==============================================================================
# 4. BOOTSTRAPPING FOR METRICS & DELONG TEST (TABLE 2)
# ==============================================================================
print("\nCalculating Table 2 Metrics (Bootstrapping 1000 iterations & DeLong Tests)...")
print("Please wait, bootstrapping 9 models x 3 datasets takes time...")


def calculate_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    lr_pos = sens / (1 - spec) if spec != 1 else np.nan
    lr_neg = (1 - sens) / spec if spec != 0 else np.nan
    auc = roc_auc_score(y_true, y_proba)
    return sens, spec, ppv, npv, lr_pos, lr_neg, auc


table_results = []

for set_name, df in datasets.items():
    print(f"\nProcessing {set_name}...")
    y_true = df['Target_AG'].values
    X_scaled = scaler.transform(df.drop(columns=['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2']))

    stacking_proba = models['Stacking'].predict_proba(X_scaled)[:, 1]

    for model_name in ordered_model_names:
        y_proba = models[model_name].predict_proba(X_scaled)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        youden_idx = np.argmax(tpr - fpr)
        opt_thresh = thresholds[youden_idx]
        y_pred = (y_proba >= opt_thresh).astype(int)

        n_bootstraps = 1000
        rng = np.random.RandomState(42)
        bootstrapped_metrics = {'Sens': [], 'Spec': [], 'PPV': [], 'NPV': [], 'LR+': [], 'LR-': [], 'AUC': []}

        for _ in range(n_bootstraps):
            indices = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[indices])) < 2: continue

            y_true_b = y_true[indices]
            y_pred_b = y_pred[indices]
            y_proba_b = y_proba[indices]

            s, sp, p, n, lr_p, lr_n, a = calculate_metrics(y_true_b, y_pred_b, y_proba_b)
            bootstrapped_metrics['Sens'].append(s)
            bootstrapped_metrics['Spec'].append(sp)
            bootstrapped_metrics['PPV'].append(p)
            bootstrapped_metrics['NPV'].append(n)
            if not np.isnan(lr_p): bootstrapped_metrics['LR+'].append(lr_p)
            if not np.isnan(lr_n): bootstrapped_metrics['LR-'].append(lr_n)
            bootstrapped_metrics['AUC'].append(a)


        def fmt_ci(data):
            if len(data) == 0: return "NA"
            mean, lower, upper = np.mean(data), np.percentile(data, 2.5), np.percentile(data, 97.5)
            return f"{mean:.2f} ({lower:.2f}-{upper:.2f})"


        z_stat, p_val = calc_delong_pvalue(y_proba, stacking_proba, y_true)
        z_str = "-" if model_name == 'Stacking' else f"{z_stat:.2f}"

        if model_name == 'Stacking':
            p_str = "-"
        elif p_val < 0.0001:
            p_str = "<0.0001"
        else:
            p_str = f"{p_val:.3f}"

        table_results.append({
            'Dataset': set_name,
            'Model': model_name,
            'Specificity (95% CI)': fmt_ci(bootstrapped_metrics['Spec']),
            'Sensitivity (95% CI)': fmt_ci(bootstrapped_metrics['Sens']),
            'PPV (95% CI)': fmt_ci(bootstrapped_metrics['PPV']),
            'NPV (95% CI)': fmt_ci(bootstrapped_metrics['NPV']),
            'LR_POS (95% CI)': fmt_ci(bootstrapped_metrics['LR+']),
            'LR_NEG (95% CI)': fmt_ci(bootstrapped_metrics['LR-']),
            'AUC (95% CI)': fmt_ci(bootstrapped_metrics['AUC']),
            'Z': z_str,
            'P value': p_str
        })

# ==============================================================================
# 5. EXPORT FINAL TABLE
# ==============================================================================
table2_df = pd.DataFrame(table_results)

# Save to CSV
table2_df.to_csv('Table_2_Master_Diagnostic_Performance.csv', index=False)
print("\n=================================================================")
print("COMPLETED! Table 2 saved as 'Table_2_Master_Diagnostic_Performance.csv'")
print("This CSV file matches the exact structure of the Q1 paper example.")
print("=================================================================")
