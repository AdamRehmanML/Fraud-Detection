import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Load the data
data = pd.read_csv('creditcardfraud/creditcard.csv')

# Test train validation split
from sklearn.model_selection import train_test_split
test_train, val = train_test_split(data, test_size=0.1, random_state=42)
train, test = train_test_split(test_train, test_size=0.2, random_state=42)

X_train, y_train = train.drop('Class', axis=1), train['Class']
X_test, y_test = test.drop('Class', axis=1), test['Class']
X_val, y_val = val.drop('Class', axis=1), val['Class']

# Check for class imbalance
class_imbalance = y_train.value_counts()[0] / y_train.value_counts()[1]

import xgboost as xbg 


# Train the model
params = {
    'objective': 'binary:logistic',  # For binary classification
    'tree_method': 'hist',          # Fast histogram optimised approximate greedy algorithm
    'device': 'cpu',           
    'learning_rate': 0.05,          # Lower learning rate for better convergence
    'max_depth': 10,                 # Depth of trees to prevent overfitting
    'n_estimators': 50000,           # Increase for better results, use early stopping
    'gamma': 10,                     # Minimum loss reduction to make a split
    'lambda': 10,                    # L2 regularisation to prevent overfitting
    'alpha': 1,                     # L1 regularisation for feature selection
    'scale_pos_weight': class_imbalance, 
    'eval_metric': 'auc'         # Scale to balance class weights
}
model = xbg.XGBClassifier(**params)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose= False)
model.save_model('xgboost_model.json')

# Predict logits
pred = model.predict(X_val)
val_probs = model.predict_proba(X_val)[:, 1]


# Evaluate the model
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix
def fbeta_score(precision, recall, beta=2.0):
    """Compute F-beta for arrays of precision, recall at each threshold."""
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-9)


precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)  # add small eps to avoid division by zero
best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]

beta = 10
fbeta_scores = fbeta_score(precision, recall, beta)
best_fbeta_index = np.argmax(fbeta_scores)
best_fbeta_threshold = thresholds[best_fbeta_index]


print(f"Best threshold for max F1 on validation: {best_threshold:.4f}")
print(f"F1 at best threshold on validation set : {f1_scores[best_index]:.4f}")

print(f"Best threshold for max F{beta} on validation: {best_fbeta_threshold:.4f}")
print(f"F{beta} at best threshold on validation set : {fbeta_scores[best_fbeta_index]:.4f}")

val_preds_custom = (val_probs >= best_threshold).astype(int)
val_preds_fbeta = (val_probs >= best_fbeta_threshold).astype(int)

confusion_f1 = confusion_matrix(y_val, val_preds_custom)
confusion_fbeta = confusion_matrix(y_val, val_preds_fbeta)

# plot confusion matrix
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.heatmap(confusion_f1, annot=True, fmt='d', cmap='Blues', ax=ax[0])
ax[0].set_title(f'F1 threshold: {best_threshold:.4f}')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('True')
sns.heatmap(confusion_fbeta, annot=True, fmt='d', cmap='Blues', ax=ax[1])
ax[1].set_title(f'F_beta={beta} threshold: {best_fbeta_threshold:.4f}')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('True')
plt.tight_layout()
plt.show()
# Save the confusion matrix
fig.savefig('confusion_matrix.png')

