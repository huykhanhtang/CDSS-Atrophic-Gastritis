import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lazypredict.Supervised import LazyClassifier
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. LOAD THE PARSIMONIOUS DATASETS (K=12 FEATURES)
# ==============================================================================
print("Loading the parsimonious datasets (Optimized K features)...")

train_df = pd.read_csv('CDSS_Web_App/Train_Final_K_Features.csv')
val_df = pd.read_csv('Val_Final_K_Features.csv')

# ==============================================================================
# 2. SEPARATE FEATURES AND TARGET
# ==============================================================================
print("Isolating features and the primary target (Target_AG)...")

y_train = train_df['Target_AG']
X_train = train_df.drop(columns=['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2'])

y_val = val_df['Target_AG']
X_val = val_df.drop(columns=['Target_AG', 'TCM_Syndromes_1', 'TCM_Syndromes_2'])

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# ==============================================================================
# 3. RUN LAZYPREDICT TO SCREEN MACHINE LEARNING ALGORITHMS
# ==============================================================================
print("\nInitializing LazyPredict to evaluate multiple algorithms simultaneously...")
print("This may take a minute depending on your CPU...")

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None, random_state=42)

models_summary, predictions = clf.fit(X_train, X_val, y_train, y_val)

# ==============================================================================
# 4. PROCESS AND EXPORT RESULTS
# ==============================================================================
print("\n✅ LazyPredict screening completed!")

pd.set_option('display.max_rows', None)

print("\nFull Rankings of All Evaluated Models on the Validation Set:")
print(models_summary[['Accuracy', 'ROC AUC', 'F1 Score', 'Time Taken']])

models_summary.to_csv('Table_S1_LazyPredict_Screening_Results.csv')
print("\n-> Saved full results to 'Table_S1_LazyPredict_Screening_Results.csv'")

# ==============================================================================
# 5. VISUALIZATION: MODEL PERFORMANCE COMPARISON
# ==============================================================================
print("Generating full performance comparison bar chart (300 DPI)...")

plt.figure(figsize=(12, 15))
sns.set_theme(style="whitegrid")

all_models = models_summary.reset_index()

ax = sns.barplot(x='ROC AUC', y='Model', data=all_models, palette='viridis')

plt.title('Complete Machine Learning Models Ranking by Validation ROC AUC',
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Area Under the ROC Curve (Validation Set)', fontsize=12)
plt.ylabel('Algorithm', fontsize=12)

for p in ax.patches:
    width = p.get_width()
    plt.text(width + 0.005, p.get_y() + p.get_height()/2. + 0.15,
             '{:1.3f}'.format(width),
             ha="left", fontsize=9) 

plt.xlim(0.5, 1.0) 
plt.tight_layout()

# Save the plot
plt.savefig('Figure_S1_LazyPredict_Full_Comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("-> Saved 'Figure_S1_LazyPredict_Full_Comparison.png'")

print("\n--------------------------------------------------")
print("✅ STEP 6 COMPLETED")
print("Next step: Select the top 4-5 core algorithms from this list to proceed to Step 7 (Hyperparameter Tuning).")
print("--------------------------------------------------")
