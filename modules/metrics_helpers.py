from modules import constants
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, accuracy_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import timeit
import itertools
from datetime import datetime
def get_classifier_metrics(y_test, y_preds):
    """
    Prints the classification report, accuracy and confusion matrix of a classification model
    """
    print(classification_report(y_test, y_preds))
    print(pd.crosstab(y_test, y_preds, rownames=["Actual Labels"], colnames=["Predicted Labels"]))
    print('Accuracy :',accuracy_score(y_test, y_preds))
    plot_conf_mat(confusion_matrix(y_test, y_preds))

def evaluate_preds(y_preds, y_true):
    """
    Performs evaluation comparison on y_true labels vs y_preds labels.
    """
    #y_preds = model.predict(X_test)

    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds)
    recall = recall_score(y_true, y_preds)
    f1 = f1_score(y_true, y_preds)
    metric_dict = {"accuracy": round(accuracy, 2), "precision": round(precision, 2), "recall": round(recall, 2), "f1": round(f1, 2)}
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")
    print('\n')
    return metric_dict
