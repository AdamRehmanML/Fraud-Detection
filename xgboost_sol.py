import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sqlite3


# Load the data
data = pd.read_csv('creditcardfraud/creditcard.csv')
conn = sqlite3.connect('creditcard.db')
data.to_sql(data, conn, if_exists='replace', index=False)
print(data.head(3))

# Test train validation split
n = int(len(data))
test, train, val = data[:n//10], data[n//10:9*n//10], data[9*n//10:]

# # Heatmap
# corr = test.corr()
# plt.figure(figsize=(12, 10))
# sns.heatmap(corr, annot=True, fmt=".2f")
# plt.show()

# XGboost classifier
import xgboost as xbg 
from xgboost import XGBClassifier

# Train the model
params = {
    'objective': 'binary:logistic',  # For binary classification
    'tree_method': 'hist',          # Use 'hist' with device='cuda' for GPU acceleration
    'device': 'cuda',               # Ensure GPU is used
    'learning_rate': 0.05,          # Lower learning rate for better convergence
    'max_depth': 20,                 # Depth of trees to prevent overfitting
    'subsample': 0.8,               # Randomly sample 80% of data for each tree
    'colsample_bytree': 0.8,        # Randomly sample 80% of features for each tree
    'n_estimators': 2000,           # Increase for better results, use early stopping
    'min_child_weight': 1,          # Minimum sum of instance weights for a leaf
    'gamma': 2,                     # Minimum loss reduction to make a split
    'lambda': 1,                    # L2 regularisation to prevent overfitting
    'alpha': 1,                     # L1 regularisation for feature selection
}
#params = {'device': 'gpu', 'tree_method': 'gpu_hist'}

model = XGBClassifier(**params)
model.fit(train.drop('Class', axis=1), train['Class'])

# Predict
pred = model.predict(test.drop('Class', axis=1))

# Evaluate the squared loss
def loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# The loss seems good however the data is imbalanced
# We need to use a metric that is robust to class imbalance
# Let's use the F1 score

from sklearn.metrics import f1_score

print(f1_score(test['Class'], pred))



print(loss(test['Class'], pred))
