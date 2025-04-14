from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=1),
        'recall': recall_score(y_test, y_pred, zero_division=1),
    }

    if len(np.unique(y_test)) > 1:
        results['roc_auc'] = roc_auc_score(y_test, y_prob)
    else:
        results['roc_auc'] = None

    return results