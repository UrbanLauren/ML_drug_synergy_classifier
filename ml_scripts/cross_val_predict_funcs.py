
# Written by Lauren Urban
# Import packages
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import (train_test_split, LeaveOneGroupOut, LeaveOneOut, RepeatedKFold,
                                     ShuffleSplit, LeavePOut, cross_validate, cross_val_score,
                                     cross_val_predict)
from sklearn.metrics import (roc_auc_score, precision_recall_curve, roc_curve, PrecisionRecallDisplay)

cv = RepeatedKFold(n_splits=3)
fig = go.Figure() # or any Plotly Express function e.g. px.bar(...)


def cross_val_predict_proba_fun(classifier, X, y):
    try:
        model = cross_val_score(classifier, X, y, cv=cv, scoring = 'accuracy')  # evaluate model
        y_preds = cross_val_predict(classifier, X, y, cv=3, method='predict')  # prediction outcomes
        y_preds_proba = cross_val_predict(classifier, X, y, cv=3, method='predict_proba') # probability for predicted outcomes
        y_preds_both = pd.DataFrame({"bug1":y_preds[:, 0], "bug2":y_preds[:, 1]}, index=y.index)
        print("y_preds_both", y_preds_both)
    except (IOError, ValueError):
        print("An I/O error or a ValueError occurred")
         
         
def plot_roc(y_true, y_score, label_name, ax):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fig.tight_layout()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.set_title(f"{label_name}: AUC = {roc_auc_score(y_true, y_score):.4f}", fontsize=10)
    
    
def precision_recall(y_true, y_predic_proba, label_name, ax):
    prec, recall, _  = precision_recall_curve(y_true, y_predic_proba)
    fig.tight_layout()
    ax.plot(recall, prec)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.set_title(f"{label_name}: AUC = {roc_auc_score(y_true, y_score):.4f}", fontsize=10)