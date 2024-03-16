
# Written by Lauren Urban
# Import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split, LeaveOneGroupOut, LeaveOneOut, RepeatedKFold,
                                     ShuffleSplit, LeavePOut, cross_validate, cross_val_score,
                                     cross_val_predict)

cv = RepeatedKFold(n_splits=3)


def cross_val_predict_proba_fun(classifier, X, y):
    try:
        model = cross_val_score(classifier, X, y, cv=cv, scoring = 'accuracy')  # evaluate model
        y_preds = cross_val_predict(classifier, X, y, cv=3, method='predict')  # prediction outcomes
        y_preds_proba = cross_val_predict(classifier, X, y, cv=3, method='predict_proba') # probability for predicted outcomes
        y_preds_both = pd.DataFrame({"bug1":y_preds[:, 0], "bug2":y_preds[:, 1]}, index=y.index)
        print("y_preds_both", y_preds_both)
    except (IOError, ValueError):
        print("An I/O error or a ValueError occurred")
         