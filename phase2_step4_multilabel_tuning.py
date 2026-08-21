import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             hamming_loss, accuracy_score, classification_report)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              VotingClassifier, StackingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
import xgboost as xgb

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD DATA & SCALER PREPARATION
# ==============================================================================
print("PHASE 2 - STEP 4: MULTI-LABEL HYPERPARAMETER TUNING & ENSEMBLE")
print("Loading parsimonious datasets (K=34 features)...")

train_df = pd.read_csv('Phase2_Train_Final_K_Features.csv')
val_df = pd.read_csv('Phase2_Val_Final_K_Features.csv')

target_tcm_cols = [col for col in train_df.columns if col.startswith('Target_TCM_')]
y_train = train_df[target_tcm_cols]
y_val = val_df[target_tcm_cols]

X_train = train_df.drop(columns=['Target_AG'] + target_tcm_cols)
X_val = val_df.drop(columns=['Target_AG'] + target_tcm_cols)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

joblib.dump(scaler, 'Phase2_CDSS_Feature_Scaler.pkl')
print(f"-> Saved scaler as 'Phase2_CDSS_Feature_Scaler.pkl'")

# ==============================================================================
# 2. CONFIGURE BASE MODELS & GRIDS (WRAPPED IN OvR)
# ==============================================================================
base_models = {
    'LogisticRegression': {
        'model': LogisticRegression(class_weight='balanced', random_state=42),
        'grid': {'estimator__C': [0.1, 1, 10], 'estimator__penalty': ['l2']}
    },
    'RandomForest': {
        'model': RandomForestClassifier(class_weight='balanced', random_state=42),
        'grid': {'estimator__n_estimators': [100, 200], 'estimator__max_depth': [None, 10]}
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'grid': {'estimator__n_estimators': [100], 'estimator__learning_rate': [0.05, 0.1]}
    },
    'SVC': {
        'model': SVC(probability=True, class_weight='balanced', random_state=42),
        'grid': {'estimator__C': [1, 10], 'estimator__kernel': ['rbf']}
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42),
        'grid': {'estimator__n_estimators': [100], 'estimator__learning_rate': [0.1, 1.0]}
    },
    'KNeighbors': {
        'model': KNeighborsClassifier(),
        'grid': {'estimator__n_neighbors': [5, 9], 'estimator__weights': ['uniform', 'distance']}
    },
    'BernoulliNB': {
        'model': BernoulliNB(),
        'grid': {'estimator__alpha': [0.1, 1.0]}
    }
}

# ==============================================================================
# 3. TRAIN AND EVALUATE 7 BASE MODELS
# ==============================================================================
print(f"\nStarting GridSearchCV for {len(base_models)} OvR Base Models...")
print("Metric: Maximizing MACRO F1-SCORE across 6 syndromes...\n")

results = []
best_estimators_dict = {}

for name, config in base_models.items():
    print(f" -> Tuning {name} (Multi-label OvR)...")

    ovr_model = OneVsRestClassifier(config['model'])

    grid_search = GridSearchCV(
        estimator=ovr_model,
        param_grid=config['grid'],
        cv=3,  
        scoring='f1_macro',
        n_jobs=-1
    )

    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    best_estimators_dict[name] = best_model.estimator  

    y_val_pred = best_model.predict(X_val_scaled)

    macro_f1 = f1_score(y_val, y_val_pred, average='macro')
    micro_f1 = f1_score(y_val, y_val_pred, average='micro')
    h_loss = hamming_loss(y_val, y_val_pred)
    subset_acc = accuracy_score(y_val, y_val_pred)

    results.append({
        'Model': name,
        'Macro_F1': macro_f1,
        'Micro_F1': micro_f1,
        'Hamming_Loss': h_loss,
        'Exact_Match_Ratio': subset_acc
    })

    joblib.dump(best_model, f'Phase2_Tuned_{name}.pkl')

# ==============================================================================
# 4. CONSTRUCT AND TRAIN 2 ENSEMBLE MODELS (OVR WRAPPED)
# ==============================================================================
print("\nConstructing Multi-label Ensemble Classifiers...")

ensemble_candidates = [
    ('lr', best_estimators_dict['LogisticRegression']),
    ('rf', best_estimators_dict['RandomForest']),
    ('svc', best_estimators_dict['SVC']),
    ('ada', best_estimators_dict['AdaBoost']),
    ('nb', best_estimators_dict['BernoulliNB'])
]

core_voting = VotingClassifier(estimators=ensemble_candidates, voting='soft', n_jobs=-1)
core_stacking = StackingClassifier(
    estimators=ensemble_candidates,
    final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
    cv=3, n_jobs=-1
)

ensembles = {
    'Soft_Voting_Ensemble': OneVsRestClassifier(core_voting),
    'Stacking_Ensemble': OneVsRestClassifier(core_stacking)
}

best_ensemble_name = None
best_ensemble_model = None
highest_macro_f1 = 0

for name, ens_model in ensembles.items():
    print(f" -> Fitting {name} (6 Independent Boards)...")
    ens_model.fit(X_train_scaled, y_train)

    y_val_pred = ens_model.predict(X_val_scaled)

    macro_f1 = f1_score(y_val, y_val_pred, average='macro')
    micro_f1 = f1_score(y_val, y_val_pred, average='micro')
    h_loss = hamming_loss(y_val, y_val_pred)
    subset_acc = accuracy_score(y_val, y_val_pred)

    results.append({
        'Model': name,
        'Macro_F1': macro_f1,
        'Micro_F1': micro_f1,
        'Hamming_Loss': h_loss,
        'Exact_Match_Ratio': subset_acc
    })

    if macro_f1 > highest_macro_f1:
        highest_macro_f1 = macro_f1
        best_ensemble_name = name
        best_ensemble_model = ens_model

joblib.dump(best_ensemble_model, 'Phase2_Final_CDSS_MultiLabel_Model.pkl')
print(f"\n Crowned '{best_ensemble_name}' as the Multi-label Champion!")
print(f"-> Saved as 'Phase2_Final_CDSS_MultiLabel_Model.pkl'")

# ==============================================================================
# 5. VISUALIZATION & EXPORT
# ==============================================================================
results_df = pd.DataFrame(results).sort_values(by='Macro_F1', ascending=False)
results_df.to_csv('Phase2_Table_S2_MultiLabel_Tuning.csv', index=False)

print("\n=================================================================")
print("MULTI-LABEL PERFORMANCE ON VALIDATION SET:")
print("=================================================================")
print(results_df.to_string(index=False))
print("=================================================================")

print("\nGenerating Figure 4: Multi-label Models Comparison...")
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

colors = ['#d62728' if 'Ensemble' in m else '#1f77b4' for m in results_df['Model']]

ax = sns.barplot(x='Macro_F1', y='Model', data=results_df, palette=colors)
plt.title('Multi-label Classification Performance (Validation Set)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Macro F1-Score (Higher is better)', fontsize=12)
plt.ylabel('Algorithm', fontsize=12)
plt.xlim(0, 1.0)

for p in ax.patches:
    width = p.get_width()
    plt.text(width + 0.01, p.get_y() + p.get_height() / 2. + 0.1,
             f'{width:.3f}', ha="left", fontsize=10)

plt.tight_layout()
plt.savefig('Phase2_Figure_4_Model_Comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Saved 'Phase2_Figure_4_Model_Comparison.png'")
print("\n PHASE 2 STEP 4 COMPLETED")
