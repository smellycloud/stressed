from modules import constants
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, accuracy_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

import itertools
from datetime import datetime

def percent_missing(df):
    """
    Returns the percent of missing values in each column in a dataframe
    """
    percent_nan = 100 * df.isnull().sum() / len(df)
    percent_nan = percent_nan.sort_values()
    return percent_nan

def nan_barplot(data):
    """
    Plots a bargraph of all features and their missing values
    """
    percent_nan = percent_missing(data)
    sns.barplot(x=percent_nan.index, y=percent_nan)
    plt.xticks(rotation=90);

def plot_conf_mat(conf_mat):
    """
    Plots a confucion matrix using Seaborn's heatmap()
    """
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(conf_mat, annot=True, cbar=True, fmt='g')
    plt.xlabel("True label")
    plt.ylabel("Predicted label")

def get_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15):
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).

  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes),
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)

  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
             horizontalalignment="center",
             color="white" if cm[i, j] > threshold else "black",
             size=text_size)

def get_feature_importance_graph(columns, model_feat_imp):
    """
    Plots features vs it's importance to the model's predictions
    """
    features = pd.DataFrame(index=columns, data=model_feat_imp, columns=['Importance'])
    sns.barplot(data=features, x=features.index, y='Importance')
    plt.xticks(rotation=90);

def plot_participant_progression(data, participant, feature):
    """
    Plots the participant's readings vs time on a lineplot
    """
    #sns.reset_defaults()
    plt.figure(figsize=(30,16), dpi=300)
    plot_data = data[data['participant'] == participant]
    plt.xlabel('Time')
    plt.ylabel('Participant '+str(participant)+' - '+feature)
    #sns.set_palette("PuBuGn_d")
    sns.lineplot(data=plot_data, x='time', y=feature, palette=['b', 'r'], hue='label')

def plot_model_performance(participant_id, metric=None):
    """
    Get a dataframe and barplot of accuracy, precision, recall, f1 and/or elapsed time
    in seconds for all models run on one participant.

    Args:
    participant_id: Participant ID
    metric: 'time' to get only time data, 'model' to get everything except time.
             If not specified, will return all metrics. Doing this will result
             in time distorting the barplot.
    Returns: None
    """
    models = participant_dictionary[participant_id]
    #print(models)

    metrics_df = pd.DataFrame(index=models.keys()).transpose()
    for model in models:
        metrics = models[model]['metrics']
        metrics = pd.DataFrame(metrics, index=metrics.keys()).iloc[0]
        metrics_df[model] = metrics
    metrics_df = metrics_df.transpose()
    metrics_df['time (sec)'] = metrics_df['time (sec)']
    if metric == 'time':
        metrics_df = metrics_df['time (sec)']
    elif metric == 'model':
        metrics_df.drop(['time (sec)'], axis=1, inplace=True)
    else:
        print('Displaying everything. Time distorts barplot.')
    display(metrics_df)
    metrics_df.plot(kind="bar", figsize=(10, 7)).legend(bbox_to_anchor=(1.0, 1.0));
    return None
