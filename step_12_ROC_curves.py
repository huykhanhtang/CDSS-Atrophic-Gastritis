import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD DATA AND SCALER
# ==============================================================================
print("Loading datasets and scaler...")

datasets = {
    'Training set': pd.read_csv('CDSS_Web_App/Train_Final_K_Features.csv'),
    'Internal validation set': pd.read_csv('Val_Final_K_Features.csv'),
    'Internal hold-out test set': pd.read_csv('Test_Final_K_Features.csv')
}

scaler = joblib.load('CDSS_Feature_Scaler.pkl')

# Extract training data to re-fit the ensembles
X_train = scaler.transform(datasets['Training set'].drop(columns=['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2']))
y_train = datasets['Training set']['Target_AG'].values

# ==============================================================================
# 2. LOAD BASE MODELS AND RECONSTRUCT ENSEMBLES
# ==============================================================================
print("Loading 7 base models and reconstructing Ensembles in memory...")

base_model_names = ['LogisticRegression', 'AdaBoost', 'BernoulliNB', 'KNeighbors', 'XGBoost', 'SVC', 'RandomForest']
models = {}
estimators_for_ensemble = []

# Load the 7 Tuned Base Models
for name in base_model_names:
    model = joblib.load(f'TunedModel_{name}.pkl')
    models[name] = model

    # We only use these 5 core models for the Ensembles (as decided in Step 8)
    if name in ['LogisticRegression', 'RandomForest', 'BernoulliNB', 'SVC', 'AdaBoost']:
        estimators_for_ensemble.append((name, model))

# Re-initialize and fit ensembles quickly
print("Fitting Soft_Voting_Ensemble and Stacking_Ensemble...")
models['Soft_Voting_Ensemble'] = VotingClassifier(estimators=estimators_for_ensemble, voting='soft', n_jobs=-1)
models['Stacking_Ensemble'] = StackingClassifier(
    estimators=estimators_for_ensemble,
    final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
    cv=5, n_jobs=-1
)

# Fit them on the training data
models['Soft_Voting_Ensemble'].fit(X_train, y_train)
models['Stacking_Ensemble'].fit(X_train, y_train)

# Arrange model order to match typical paper presentations
ordered_model_names = base_model_names + ['Soft_Voting_Ensemble', 'Stacking_Ensemble']

# ==============================================================================
# 3. GENERATE FIGURE WITH 3 SUBPLOTS
# ==============================================================================
print("Generating ROC Curves Subplots (300 DPI)...")

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

# Modern color palette for 9 lines
colors = sns.color_palette("husl", 9)
color_map = {name: colors[i] for i, name in enumerate(ordered_model_names)}

for idx, (set_name, df) in enumerate(datasets.items()):
    ax = axes[idx]
    y_true = df['Target_AG'].values
    X_scaled = scaler.transform(df.drop(columns=['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2']))

    for model_name in ordered_model_names:
        y_proba = models[model_name].predict_proba(X_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        # Emphasize the Stacking model with a thicker, solid line
        lw = 3.0 if 'Stacking' in model_name else 1.5
        ls = '-' if 'Ensemble' in model_name else '--'

        # Clean up names for the legend (e.g., 'Stacking_Ensemble' -> 'Stacking')
        display_name = model_name.replace('_Ensemble', '').replace('_', ' ')

        ax.plot(fpr, tpr, color=color_map[model_name], lw=lw, linestyle=ls,
                label=f'{display_name} (AUC = {auc:.2f})')

    ax.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.5)
    ax.set_xlabel('1 - Specificity', fontsize=13)
    ax.set_ylabel('Sensitivity', fontsize=13)
    ax.legend(loc='lower right', fontsize=9, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Figure_3_Step12_ROC_Curves_Subplots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved 'Figure_3_Step12_ROC_Curves_Subplots.png'")
