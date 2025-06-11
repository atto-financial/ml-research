# Set up
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sqlalchemy import create_engine
import joblib

# Connect MySQL
engine = create_engine('mysql+mysqlconnector://root:fPQaiBKOdemHsgYaQlFKbqeLkAmjEpMl@junction.proxy.rlwy.net:58810/railway')

# Query
query = """
SELECT set_1 AS set1, set_2 AS set2, set_3 AS set3, set_4 AS set4, 
       set_5 AS set5, set_6 AS set6, set_7 AS set7, set_8 AS set8, 
       set_9 AS set9, set_10 AS set10, set_11 AS set11, 
       init_payoff_score AS ins, user_status AS ust 
FROM users 
WHERE user_status = 1 OR (user_status = 0 and payoff_score > 4)
"""
raw_dat = pd.read_sql(query, engine)

# Data display
print("Data:")
print(raw_dat.head())
print("Number of data:", raw_dat.shape)
print("Class distribution:")
print(raw_dat["ust"].value_counts(normalize=True))

# rearrage data
X = raw_dat.drop(["ust", "ins"], axis=1)
y = raw_dat["ust"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("Model trained on training data")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("Tested model on testing data")

# Test results
print("Test Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# Cross-validation
print("Cross-validation")
cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
print("Cross-validated ROC AUC (mean):", cv_scores.mean())
print("Cross-validated ROC AUC (all folds):", cv_scores)

# Prediction by Cross-validation
y_pred_cv = cross_val_predict(model, X, y, cv=5, method='predict')
y_prob_cv = cross_val_predict(model, X, y, cv=5, method='predict_proba')

# Result
print("Cross-validation test")
print("CV Accuracy:", accuracy_score(y, y_pred_cv))
print("CV Precision:", precision_score(y, y_pred_cv))
print("CV Recall:", recall_score(y, y_pred_cv))
print("CV ROC AUC:", roc_auc_score(y, y_prob_cv[:, 1]))

# Feature Importance 
model.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances
}).sort_values('importance', ascending=False)
print("Feature Importance:\n", feature_importance)


# Save model 
joblib.dump(model, 'mlrf_model_X.pkl')
print("Save model complete")