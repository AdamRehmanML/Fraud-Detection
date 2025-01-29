# Fraud-Detection

This repository contains experiments and code for detecting financial fraud in credit card transactions using various machine learning methods. The primary dataset comes from [Kaggle's credit card fraud dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).

## Overview

1. **Data**:  
   - The dataset consists of transactions labeled as fraudulent or legitimate.  
   - Strong class imbalance (legitimate transactions vastly outnumber fraudulent ones).

2. **Modeling**:  
   - The primary model in this repository is an **XGBoost** (Gradient Boosted Decision Trees) classifier.  
   - Configuration includes parameters such as `max_depth`, `learning_rate`, `n_estimators`, `gamma`, regularization terms (`alpha`, `lambda`), and `scale_pos_weight` to address class imbalance.  
   - We use a **validation set** to monitor performance and apply **early stopping** or threshold tuning.

3. **Evaluation Metrics**:  
   - We calculate **Precision** and **Recall**, then derive the **F1 Score** (harmonic mean of precision and recall).  
   - We also compute **Fβ** (with β > 1, e.g., β=10) to emphasize **Recall** if missing fraud is costlier.  
   - We select thresholds that maximize either **F1** or **Fβ**, then compare confusion matrices.

4. **Threshold Tuning**:  
   - The default prediction threshold for binary classifiers is 0.5.  
   - Because of class imbalance, we tune the threshold by examining the **Precision-Recall** curve and picking a value that maximizes a specific F-measure.

5. **Confusion Matrices**:  
   - We visualize confusion matrices side-by-side for the **F1-optimized** threshold vs. the **Fβ-optimized** threshold.  
   - This helps illustrate the trade-offs in **false positives** vs. **false negatives** for different emphasis on recall.

---

## Dataset Imports

1. **Kaggle CLI**:  
   ```bash
   kaggle datasets download mlg-ulb/creditcardfraud
