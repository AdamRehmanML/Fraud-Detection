import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the data
data = pd.read_csv('creditcard.csv')
print("Showing first record of data:")
print(data.head(1))

# Test train validation split
from sklearn.model_selection import train_test_split
test_train, val = train_test_split(data, test_size=0.1, random_state=42)
train, test = train_test_split(test_train, test_size=0.2, random_state=42)

X_train, y_train = train.drop('Class', axis=1), train['Class']
X_test, y_test = test.drop('Class', axis=1), test['Class']
X_val, y_val = val.drop('Class', axis=1), val['Class']

# Check for class imbalance
class_imbalance = y_train.value_counts()[0] / y_train.value_counts()[1]

import xgboost as xgb  # Fixed typo: 'xbg' -> 'xgb'

# Train the model
params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'device': 'cpu',
    'learning_rate': 0.05,
    'max_depth': 10,
    'n_estimators': 500,
    'gamma': 10,
    'lambda': 10,
    'alpha': 1,
    'scale_pos_weight': class_imbalance,
    'eval_metric': 'auc'
}
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
model.save_model('xgboost_model.json')
# Predict probabilities
val_probs = model.predict_proba(X_val)[:, 1]

# Evaluate the model
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix

def fbeta_score(precision, recall, beta=2.0):
    """Compute F-beta for arrays of precision, recall at each threshold."""
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-9)

# Compute precision, recall, and thresholds once
precision, recall, thresholds = precision_recall_curve(y_val, val_probs)

# F1 score (for reference, optional)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]
print(f"Best threshold for max F1: {best_threshold:.4f}")
print(f"F1 at best threshold: {f1_scores[best_index]:.4f}")

# Loop over beta values from 1 to 20
for beta in range(1, 21):
    # Compute F-beta scores for this beta
    fbeta_scores = fbeta_score(precision, recall, beta)
    best_fbeta_index = np.argmax(fbeta_scores)
    best_fbeta_threshold = thresholds[best_fbeta_index]

    # Predictions with the best F-beta threshold
    val_preds_fbeta = (val_probs >= best_fbeta_threshold).astype(int)
    confusion_fbeta = confusion_matrix(y_val, val_preds_fbeta)

    # Create and save the plot
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_fbeta, annot=True, fmt='d', cmap='Blues')
    plt.title(f'F_beta={beta} threshold: {best_fbeta_threshold:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    # Save the plot (filename includes beta value)
    plt.savefig(f'F_beta_{beta}.jpg')
    #plt.close()  # Close the figure to free memory

    # Optional: Print results
    print(f"F_beta={beta}: Best threshold = {best_fbeta_threshold:.4f}, Score = {fbeta_scores[best_fbeta_index]:.4f}")

