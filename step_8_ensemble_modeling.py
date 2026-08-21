import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_curve, roc_auc_score, recall_score,
                             confusion_matrix, accuracy_score, precision_score,
                             f1_score, log_loss)

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD DATA AND SCALER
# ==============================================================================
print("Loading datasets and the saved Feature Scaler...")

train_df = pd.read_csv('CDSS_Web_App/Train_Final_K_Features.csv')
val_df = pd.read_csv('Val_Final_K_Features.csv')

y_train = train_df['Target_AG']
X_train = train_df.drop(columns=['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2'])

y_val = val_df['Target_AG']
X_val = val_df.drop(columns=['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2'])

scaler = joblib.load('CDSS_Feature_Scaler.pkl')
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ==============================================================================
# 2. LOAD TUNED BASE MODELS (FROM STEP 7)
# ==============================================================================
print("Loading the Top 5 Tuned Base Models to form the Medical Board...")

model_names = ['LogisticRegression', 'AdaBoost', 'BernoulliNB', 'RandomForest', 'SVC']
estimators = []

for name in model_names:
    model = joblib.load(f'TunedModel_{name}.pkl')
    estimators.append((name, model))

print(f"Successfully loaded {len(estimators)} elite base models.")

# ==============================================================================
# 3. CONSTRUCT ENSEMBLE MODELS
# ==============================================================================
print("\nConstructing Ensemble Classifiers...")

voting_clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
    cv=5,
    n_jobs=-1
)

ensembles = {
    'Soft_Voting_Ensemble': voting_clf,
    'Stacking_Ensemble': stacking_clf
}

# ==============================================================================
# 4. TRAIN, EVALUATE AND COMPARE ALL MODELS
# ==============================================================================
print("\nTraining and evaluating models on the Validation Set...")

results = []
roc_data = {}  # To store data for plotting Figure 5

for name, model in estimators:
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_val_proba)
    roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc_score(y_val, y_val_proba)}

best_ensemble_name = None
best_ensemble_model = None
highest_auc = 0

for name, ensemble_model in ensembles.items():
    print(f"-> Fitting {name}...")
    ensemble_model.fit(X_train_scaled, y_train)

    y_val_pred = ensemble_model.predict(X_val_scaled)
    y_val_proba = ensemble_model.predict_proba(X_val_scaled)[:, 1]

    val_auc = roc_auc_score(y_val, y_val_proba)
    val_recall = recall_score(y_val, y_val_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred)
    val_logloss = log_loss(y_val, y_val_proba)

    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
    val_specificity = tn / (tn + fp)

    results.append({
        'Model': name,
        'Val_Recall': val_recall,
        'Val_Specificity': val_specificity,
        'Val_Accuracy': val_acc,
        'Val_Precision': val_precision,
        'Val_F1': val_f1,
        'Val_ROC_AUC': val_auc,
        'Val_LogLoss': val_logloss
    })

    fpr, tpr, _ = roc_curve(y_val, y_val_proba)
    roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': val_auc}

    if val_auc > highest_auc:
        highest_auc = val_auc
        best_ensemble_name = name
        best_ensemble_model = ensemble_model

joblib.dump(best_ensemble_model, 'Final_CDSS_Screening_Model.pkl')
print(f"\n Saved '{best_ensemble_name}' as 'Final_CDSS_Screening_Model.pkl' for Step 9.")

# ==============================================================================
# 5. VISUALIZATION: ROC CURVES (FIGURE 5)
# ==============================================================================
print("\nGenerating Figure 5: ROC Curves Comparison...")

plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")

colors = {
    'LogisticRegression': '#2ca02c',  
    'AdaBoost': '#17becf',            
    'RandomForest': '#1f77b4',        
    'BernoulliNB': '#9467bd',        
    'SVC': '#8c564b',                 
    'Soft_Voting_Ensemble': '#ff7f0e',
    'Stacking_Ensemble': '#d62728'    
}

for name, data in roc_data.items():
    lw = 2.5 if 'Ensemble' in name else 1.5
    ls = '-' if 'Ensemble' in name else '--'
    plt.plot(data['fpr'], data['tpr'], color=colors[name], lw=lw, linestyle=ls,
             label=f"{name} (AUC = {data['auc']:.3f})")

plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle=':')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curves:\nTop 5 Base Models vs. Ensembles',
          fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=10, frameon=True, shadow=True)
plt.tight_layout()

plt.savefig('Figure_5_ROC_Curves_Comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Saved 'Figure_5_ROC_Curves_Comparison.png'")

results_df = pd.DataFrame(results)
print("\n=================================================================")
print("ENSEMBLE MODELS PERFORMANCE ON VALIDATION SET:")
print("=================================================================")
pd.set_option('display.max_colwidth', 50)
print(results_df[['Model', 'Val_Recall', 'Val_Specificity', 'Val_Accuracy', 'Val_ROC_AUC']].to_string(index=False))
print("-----------------------------------------------------------------")
