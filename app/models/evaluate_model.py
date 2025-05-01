from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_val_score
import pandas as pd
    
def test_set(model, X_test, y_test):
    # Test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    results = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=1),
        "test_recall": recall_score(y_test, y_pred, zero_division=1),
        "test_roc_auc": roc_auc_score(y_test, y_prob[:, 1])
    }
    return results   
    
    
def cross_validation(model, X_train, X_test, y_train, y_test, X, y, scoring='roc_auc'):
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=3, scoring='roc_auc')
    # Prediction by Cross-validation
    y_pred_cv = cross_val_predict(model, X, y, cv=3, method='predict')
    y_prob_cv = cross_val_predict(model, X, y, cv=3, method='predict_proba')
    results = {
            f"cross_validated_{scoring.upper()}_mean": cv_scores.mean(),
            f"cross_validated_{scoring.upper()}_all_folds": cv_scores.tolist(),
            
            "cross_validated_accuracy": accuracy_score(y, y_pred_cv),
            "cross_validated_precision": precision_score(y, y_pred_cv, zero_division=1),
            "cross_validated_recall": recall_score(y, y_pred_cv, zero_division=1),
            "cross_validated_roc_auc": roc_auc_score(y, y_prob_cv[:, 1])
    }
    return results
    

def features_importance(model, X, y = None):
    feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
    return feature_importance