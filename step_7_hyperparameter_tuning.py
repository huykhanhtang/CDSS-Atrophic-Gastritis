import pandas as pd
import numpy as np
import warnings
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, accuracy_score, precision_score, f1_score, log_loss

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
import xgboost as xgb

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD DATA AND PREPARE FEATURES
# ==============================================================================
print("Loading parsimonious datasets (K=9 features)...")

train_df = pd.read_csv('CDSS_Web_App/Train_Final_K_Features.csv')
val_df = pd.read_csv('Val_Final_K_Features.csv')

y_train = train_df['Target_AG']
X_train = train_df.drop(columns=['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2'])

y_val = val_df['Target_AG']
X_val = val_df.drop(columns=['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2'])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

joblib.dump(scaler, 'CDSS_Feature_Scaler.pkl')
print("-> Saved feature scaler as 'CDSS_Feature_Scaler.pkl'")

# ==============================================================================
# 2. DEFINE MODELS AND HYPERPARAMETER GRIDS
# ==============================================================================
models_and_grids = {
    'LogisticRegression': {
        'model': LogisticRegression(class_weight='balanced', random_state=42),
        'grid': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(class_weight='balanced', random_state=42),
        'grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'scale_pos_weight': [1, 2, 3]  # Helps with recall optimization
        }
    },
    'SVC': {
        'model': SVC(probability=True, class_weight='balanced', random_state=42),
        'grid': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42),
        'grid': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.5, 1.0]
        }
    },
    'KNeighbors': {
        'model': KNeighborsClassifier(),
        'grid': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # Manhattan vs Euclidean distance
        }
    },
    'BernoulliNB': {
        'model': BernoulliNB(),
        'grid': {
            'alpha': [0.1, 0.5, 1.0, 2.0]  # Laplace smoothing
        }
    }
}

# ==============================================================================
# 3. EXECUTE GRID SEARCH CV (OPTIMIZING FOR RECALL)
# ==============================================================================
print(f"\nStarting GridSearchCV for {len(models_and_grids)} models...")
print("Objective: Maximize RECALL (Sensitivity) to minimize False Negatives in Screening.\n")

results = []
best_estimators = {}

for name, config in models_and_grids.items():
    print(f"Tuning {name}...")

    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['grid'],
        cv=5,
        scoring='recall',
        n_jobs=-1
    )

    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    best_estimators[name] = best_model

    y_val_pred = best_model.predict(X_val_scaled)
    y_val_proba = best_model.predict_proba(X_val_scaled)[:, 1]

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
        'Best_Params': str(grid_search.best_params_),
        'Val_Recall': val_recall,
        'Val_Specificity': val_specificity,
        'Val_Accuracy': val_acc,
        'Val_Precision': val_precision,
        'Val_F1': val_f1,
        'Val_ROC_AUC': val_auc,
        'Val_LogLoss': val_logloss
    })

    model_filename = f'TunedModel_{name}.pkl'
    joblib.dump(best_model, model_filename)

# ==============================================================================
# 4. SUMMARIZE AND EXPORT RESULTS
# ==============================================================================
results_df = pd.DataFrame(results).sort_values(by=['Val_Recall', 'Val_ROC_AUC'], ascending=[False, False])

print("\n=================================================================")
print("✅ TUNING COMPLETED. FINAL PERFORMANCE ON VALIDATION SET:")
print("=================================================================")
# Display the table clearly
pd.set_option('display.max_colwidth', 50)
print(results_df[['Model', 'Val_Recall', 'Val_Specificity', 'Val_Accuracy', 'Val_Precision', 'Val_F1', 'Val_ROC_AUC', 'Val_LogLoss']].to_string(index=False))

# Export to CSV for manuscript
results_df.to_csv('Table_S2_Hyperparameter_Tuning_Results.csv', index=False)
print("\n-> Saved detailed tuning results to 'Table_S2_Hyperparameter_Tuning_Results.csv'")
print("-> Saved all 7 optimal models as .pkl files for future Ensemble integration.")
print("-----------------------------------------------------------------")
