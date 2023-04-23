#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, accuracy_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

import timeit
import itertools
from datetime import datetime


# # Constants

# In[2]:


RANDOM_STATE = 41
TRAIN_PARTICIPANT_PROPORTION = 0.8
SLICE_TEST_PROPORTION = 0.35
WINDOW_SIZE = 100
CV_FOLDS = 5
RSCV_VERBOSITY = 1

n_participants = 35

directory = 'Stress-Predict-Dataset-main/Raw_data'

# For reference
participant_dictionary = {
    0: {
        'Dummy Classifier': {
            'model': None,
            'metrics': None,
            'time': None
        }
    }
}


# # Helper Functions

# In[3]:


def percent_missing(df):
    """
    Returns the percent of missing values in each column in a dataframe
    """
    percent_nan = 100 * df.isnull().sum() / len(df)
    percent_nan = percent_nan.sort_values()
    return percent_nan


# In[4]:


def nan_barplot(data):
    """
    Plots a bargraph of all features and their missing values
    """
    percent_nan = percent_missing(data)
    sns.barplot(x=percent_nan.index, y=percent_nan)
    plt.xticks(rotation=90);


# In[5]:


def plot_conf_mat(conf_mat):
    """
    Plots a confucion matrix using Seaborn's heatmap()
    """
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(conf_mat, annot=True, cbar=True, fmt='g')
    plt.xlabel("True label")
    plt.ylabel("Predicted label")


# In[6]:


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


# In[7]:


def get_feature_importance_graph(columns, model_feat_imp):
    """
    Plots features vs it's importance to the model's predictions
    """
    features = pd.DataFrame(index=columns, data=model_feat_imp, columns=['Importance'])
    sns.barplot(data=features, x=features.index, y='Importance')
    plt.xticks(rotation=90);


# In[8]:


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


# In[9]:


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


# In[10]:


def get_participant_data(model_data, participant_id):
    """
    Args:
    model_data: Dictionary with all participant dataframes (dict)
    participant_id: Participant ID (int)

    Returns:
    A dataframe corresponding to an individual participant
    """
    return model_data[participant_id]['data']


# In[11]:


def parse_data_to_df(participant_number, key, has_refresh, columns):
    '''
    Parse the raw sensor .csv file into a usable dataframe

    Args:
    participant_id: Participant ID (int)
    key: Sensor name
    has_refresh: True if sensor data has a refresh rate
    columns: Dict of columns expected in the output dataframe

    Returns:
    A dataframe with individual sensor data scaled down to 1Hz if necessary
    '''
    df = pd.read_csv(directory+'/S'+str(participant_number).zfill(2)+'/'+key+'.csv', header=None)
    # Store initial timestamp
    time = int(df.iloc[0][0])
    if has_refresh:
        # Store sample rate of sensor. The values obtained are averaged in chunks of
        #length sample_rate and stored as a singular row
        sample_rate = int(df.iloc[1][0])
        # Drop sensor and timestamp information from the original dataframe
        df.drop(index=[0, 1], axis=0, inplace=True)
        # Rename columns according to the specification in the features dict
        df.rename(columns=columns, inplace=True)
        for c in columns.values():
            # Mean of values of length sample_rate
            df[c] = df.groupby(df.index // sample_rate)[c].transform('mean')
        df = df.iloc[::int(sample_rate), :]
    else:
        # Condition to handle data without refresh rate
        df.rename(columns=columns, inplace=True)
        df.drop(index=[0], axis=0, inplace=True)
        if key == 'IBI':
            # Special case to handle IBI_xxx_xxx
            df['IBI_DETECT_TIME'] += time
            #df['IBI_DETECT_TIME'] = pd.to_numeric(df['IBI_DETECT_TIME'], downcast='integer')
            df['IBI_DETECT_TIME'] = df['IBI_DETECT_TIME'].round()
            df['IBI_DETECT_TIME'] = df['IBI_DETECT_TIME'].astype(int)
            df.rename(columns={'IBI_DETECT_TIME': 'time'}, inplace=True)
            df.reset_index(drop=True, inplace=True)
            #print(df)
            # IMPORTANT: Fill in missing values
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            return df
    df.reset_index(drop=True, inplace=True)
    # Append timestamps to the dataframe starting from the initial timestamp to timestamp+len(data)
    df['time'] = range(time-1, time+len(df)-1)
    return df


# In[12]:


invalid_participant_id = set()
def append_labels(data, participant_number):
    '''
    Append the target feature to the dataframe by referencing the timestamp file
    Args:
    data: Participant sensor data
    participant_number: Participant ID (int)
    Returns:
    A dataframe with stress label appended. 1 if stressed, 0 if not
    '''
    # Initialise all label (target variable) values to 0
    data['label'] = 0
    tags = pd.read_csv(directory+'/S'+str(participant_number).zfill(2)+'/tags_S'+str(participant_number).zfill(2)+'.csv', header=None)
    tags[0] = tags[0].round().astype(int)
    # Handle cases where timestamp dataframe has 9 and 7 rows
    if len(tags) == 9:
        # experiment name : [start timestamp index, end timestamp index, expected experiment duration + buffer duration (experiments might take a bit longer than specified) in mins]
        # 3 mins was chosen arbitrarily. Most experiments do not exceed 2 minutes from the expected duration
        stress_indices = {'stroop': [0, 1, 5+3], 'tsst': [2, 3, 10+3], 'hyperventilation': [4, 5, 2+1]}
    elif len(tags) == 7:
        stress_indices = {'stroop': [0, 1, 5+3], 'tsst': [2, 3, 10+3], 'hyperventilation': [4, 5, 2+1]}
    else:
        print('Unable to determine labels for patricipant', participant_number, '\n')
        invalid_participant_id.add(participant_number)
        print('-'*100)
        print('\n')
        return None
    for stressor in stress_indices:
            start = tags[0][stress_indices[stressor][0]]
            end = tags[0][stress_indices[stressor][1]]-1
            try:
                start_index = np.where(data['time'] == start)[0][0]
                end_index = np.where(data['time'] == end)[0][0]
            except:
                print('EXCLUDE PARTICIPANT '+str(participant_number)+'. INVALID TIMESTAMP')
                print('-'*100)
                print('\n')
                return None
            finally:
                difference = abs((float(end)-float(start))/(60))
                print('participant', participant_number, stressor, start, end, '%.3f'%difference, 'minutes')
                if difference > stress_indices[stressor][2]:
                    print('EXCLUDE PARTICIPANT '+str(participant_number)+'. INCORRECT TEST DURATION DETECTED')
                    invalid_participant_id.add(participant_number)
                    print('-'*100)
                    print('\n')
                    return None

            print(start_index, end_index)
            #if difference > stress_indices[stressor][2]:
                #print('EXCLUDE PARTICIPANT '+str(participant_number)+'. INCORRECT TEST DURATION DETECTED')
                #invalid_participant_id.add(participant_number)
            #indices = np.asarray(data.index[(data['time'] >= start) & (data['time'] <= end)].tolist())
            #print(indices)

            # Set stress test labels to 1
            data.loc[start_index:end_index, 'label'] = 1
    print('-'*100)
    print('\n')
    return data


# In[13]:


def build_dataframe(participant_number):
    '''
    Merge all participant features into a single dataframe
    Args:
    data: Participant sensor data
    Returns:
    A dataframe with all sensor data merged by timestamp
    '''
    if participant_number < 1 or participant_number > n_participants:
        print('Invalid participant number')
        return None
    else:
        data = pd.DataFrame()
        count = 0
        for feature in features:
            df = parse_data_to_df(participant_number, feature, features[feature]['has_refresh'], features[feature]['columns'])
            if count == 0:
                data = df.copy(deep=True)
            else:
                data = data.merge(df, on='time', how='outer')
            count += 1
        data.insert(0, 'participant', participant_number)
    return data


# In[14]:


def create_participant_dict():
    """
    Returns:
    A dictionary with all participant dataframes
    """
    participant_data = dict()
    for i in range(2, n_participants+1):
        extracted_data = append_labels(build_dataframe(i), i)
        if extracted_data is not None:
            extracted_data.reset_index(drop=True, inplace=True)
            participant_data[i] = {'data': None}
            participant_data[i]['data'] = extracted_data
    return participant_data


# In[15]:


def slice_train_test(data, test_proportion=SLICE_TEST_PROPORTION, chunk_size=WINDOW_SIZE):
    """
    Continuously grabs 20 second and 80 second snapshots for every 100 seconds and splits them
    into test and train dataframes respectively. Order is maintained.

    Args:
    data: Participant dataframe
    test_proportion: Proportion of data to use for testing (default: 0.2)
    chunk_size: Size of chunks to split into train and test (default: 100).
                If unspecified/out of range, will automatically split into 10 equal chunks.
    Returns:
    train, test : A tuple with train and test data
    """
    if chunk_size < 2:
        print('Invalid chunk_size. Has to be a minimum of 2')
        return None, None
    if data is None:
        print('No data provided')
        return None, None
    if test_proportion >= 1 or test_proportion <= 0:
        print('Invalid test proportion')
        return None, None

    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    n_rows = data.shape[0]
    print('Total number of rows:', n_rows)
    train = pd.DataFrame()
    test = pd.DataFrame()

    if chunk_size > n_rows:
        chunk_size = int(n_rows // 10)
        print('Chunk size > Number of rows. Setting chunk size to', chunk_size)
    count = 0
    start_index = 0

    len_test_slice = (abs(int(test_proportion*100)) / 100) * chunk_size
    len_train_slice = abs(int(chunk_size - len_test_slice))
    #print(len_train_slice)
    print('Splitting data into train-test slices',chunk_size,'rows at a time')
    print('Train size:',len_train_slice,'\nTest size:',chunk_size-len_train_slice)
    for index in range(0, n_rows, chunk_size):
        if index == 0:
            continue
        else:
            count += 1
            data_slice = data.iloc[start_index: chunk_size + start_index]
            data_slice.reset_index(drop=True, inplace=True)

            train_slice = data_slice.iloc[:len_train_slice]
            test_slice = data_slice.iloc[len_train_slice:]
            start_index = index

            train = pd.concat([train, train_slice], ignore_index=True)
            test = pd.concat([test, test_slice], ignore_index=True)

            n_leftovers = len(data) - (len(train)+len(test))
    #print('Iter count:', count)
    if n_leftovers > 0:
        print('Leftover rows:', n_leftovers, '\nMoving into train set...\n')
        leftovers = data[len(data)-n_leftovers:]
        train = pd.concat([train, leftovers], ignore_index=True)
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    print('Train value counts: \n',train['label'].value_counts())
    print('Test value counts: \n',test['label'].value_counts())
    print('-'*100)

    # Sanity check!
    if len(train) + len(test) == n_rows:
        return train, test
    else:
        print('Row count mismatch!')
        return None, None


# In[30]:


def slice_train_test1(data, buffer_size=2, drop_buffer=True, test_proportion=SLICE_TEST_PROPORTION, window_size=WINDOW_SIZE):
    """
    Continuously grabs 20 second and 80 second snapshots for every 100 seconds and splits them at random
    into test and train dataframes respectively. Order is maintained.

    Args:
    data: Participant dataframe
    test_proportion: Proportion of data to use for testing (default: 0.2)
    window_size: Size of chunks to split into train and test (default: 100).
                If unspecified/out of range, will automatically split into 10 equal chunks.
    Returns:
    train, test, valid : A tuple with train and test data
    """
    np.random.seed(RANDOM_STATE)
    # Edge case check
    if data is None:
        print('No data provided')
        return None, None
    if test_proportion >= 1 or test_proportion <= 0:
        print('Invalid test proportion')
        return None, None
    
    # Fill in nans if not already done
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # Prep output dataframes
    n_rows = data.shape[0]
    print('Total number of rows:', n_rows)
    train = pd.DataFrame()
    test = pd.DataFrame()
    buffer = pd.DataFrame()
    
    # Deal with wrong input
    if window_size > n_rows or window_size < (buffer_size*2)+1:
        window_size = int(n_rows // 10)
        print('Invalid window size. Setting window size to', window_size)

    print('Window size:', window_size)
    test_slice_size = int(test_proportion * WINDOW_SIZE)
    print('Test slice size:', test_slice_size)

    count = 0
    start_index = 0
    
    # Slice window_size sized chunks from dataframe
    for index in range(1, n_rows, window_size):
        # Prevent overflow
        if n_rows - index < window_size:
            break
        end = index + window_size
        # Slice window
        data_slice = data.iloc[index:end]
        data_slice.reset_index(drop=True, inplace=True)
        while True:
                # Pick indices of random test slices
                picked_top = int(np.random.uniform(buffer_size, len(data_slice) - buffer_size))
                #picked_top = int(np.random.uniform(0, len(data_slice)))
                picked_bottom = picked_top + test_slice_size
                buffer_top = picked_top - buffer_size
                buffer_bottom = picked_bottom + buffer_size
                # Prevent slicing from where there is no space for buffer
                if buffer_top > 0 and buffer_bottom < len(data_slice):
                    break
        #print('Index:', index)
        #print('Top:', 0)
        #print('Buffer top:', buffer_top)
        #print('Picked test top:', picked_top)
        #print('Picked test bottom:', picked_bottom)
        #print('Buffer bottom:', buffer_bottom)
        #print('Bottom:', len(data_slice))
        #print('\n')
        if drop_buffer is True:
            # Slice train, test and buffer
            train = pd.concat([train, data_slice[0:buffer_top]], ignore_index=True)
            buffer_top_data = data_slice[buffer_top:picked_top]
            buffer = pd.concat([buffer, buffer_top_data], ignore_index=True)

            picked_test = data_slice[picked_top:picked_bottom]
            test = pd.concat([test, picked_test], ignore_index=True)

            buffer_bottom_data = data_slice[picked_bottom:buffer_bottom]
            buffer = pd.concat([buffer, buffer_bottom_data], ignore_index=True)

            train = pd.concat([train, data_slice[buffer_bottom:index+window_size]], ignore_index=True)
            start_index = index
        else:
            print('TODO')
            return None, None

    # Move leftover rows into training set
    n_leftovers = n_rows - (len(buffer) + len(train) + len(test))
    if n_leftovers > 0:
        print('Leftover rows:', n_leftovers, '\nMoving into train set...\n')
        leftovers = data[n_rows-n_leftovers:]
        train = pd.concat([train, leftovers], ignore_index=True)

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    buffer.reset_index(drop=True, inplace=True)
    print(len(train), len(buffer), len(test))
    # Sanity check
    if len(train) + len(buffer) + len(test) == n_rows:
        print('Row count OK!')
        print('Train value counts: \n',train['label'].value_counts())
        print('Test value counts: \n',test['label'].value_counts())
        print(len(buffer), 'rows lost to buffer')
        print('-'*100)
        return train, test, buffer

    print('Row count mismatch!')
    print('-'*100)
    return None, None


# In[17]:


def get_classifier_metrics(y_test, y_preds):
    """
    Prints the classification report, accuracy and confusion matrix of a classification model
    """
    print(classification_report(y_test, y_preds))
    print(pd.crosstab(y_test, y_preds, rownames=["Actual Labels"], colnames=["Predicted Labels"]))
    print('Accuracy :',accuracy_score(y_test, y_preds))
    plot_conf_mat(confusion_matrix(y_test, y_preds))


# In[18]:


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


# In[28]:


def run_model(model_dict=None, model_data=None, participant_id=None, scale=False):
    """
    Train and test the provided model after performing all necessary preprocessing.
    Automatically performs hyperparameter tuning for all applicable models.

    Args:
    model_dict: Dictionary with model object and hyperparameter grid
    model_data: Dictionary with all participant data
    participant_id: Participant ID
    scale: Is standardization required? True if Yes, False by default
    Returns:
    best_model, metrics_dict : A tuple with the best model and it's corresponding performance
                               metrics dictionary
    """
    if model_dict == None or model_data == None or participant_id == None:
        print('Invalid arguments')
        return None
    model = None
    params = None
    rscv = None
    participant_data = model_data[participant_id]['data']
    model = model_dict['model']
    params = model_dict['params']
    dropped_features = ['label', 'time', 'participant', 'BVP']

    print('='*100)
    print("PARTICIPANT", participant_id)
    print('='*100)

    #train, test = slice_train_test(participant_data, SLICE_TEST_PROPORTION, SLICE_CHUNK_SIZE)
    train, test, valid = slice_train_test1(participant_data, buffer_size=5, test_proportion=SLICE_TEST_PROPORTION, window_size=WINDOW_SIZE)
    X_train = train.drop(dropped_features, axis=1)
    X_test = test.drop(dropped_features, axis=1)
    y_train = train['label']
    y_test = test['label']

    if scale == True:
        print('Scaling data...')
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    rscv = RandomizedSearchCV(model, scoring='accuracy', cv=CV_FOLDS,
                              param_distributions=params, refit=True,
                              n_jobs=-1, random_state=RANDOM_STATE,
                              return_train_score=True, verbose=RSCV_VERBOSITY)
    print('Running RandomizedSearchCV')
    rscv.fit(X_train, y_train)
    print('Model scorer:', rscv.scorer_)
    best_model = rscv
    print('Best params:', rscv.best_params_)
    rscv_results = pd.DataFrame(rscv.cv_results_)
    print('RSCV: Mean Train Score:',np.mean(rscv_results['mean_train_score']))
    print('RSCV: Mean Test Score:',np.mean(rscv_results['mean_test_score']))
    #display(rscv_results)
    result = X_test

    y_preds = best_model.predict(X_test)

    #display(y_probs_positive)

    print(classification_report(y_test, y_preds))
    metrics_dict = evaluate_preds(y_preds, y_test)
    #plot_conf_mat(confusion_matrix(y_test, y_preds))
    print(confusion_matrix(y_test, y_preds))
    #get_feature_importance_graph(X_test.columns, best_model.feature_importances_)
    #predtest = pd.DataFrame([y_test, y_preds])
    get_confusion_matrix(y_true=y_test, y_pred=y_preds, figsize=(5,5))

    return best_model, metrics_dict


# In[20]:


def run_participants(participants_subset, classifier, scale=False):
    """
    Run an ML model for a subset of participants

    Args:
    participants_subset: List with participant IDs to run model on
    classifier: String with classifier ID. Refer classifier_dictionary
    scale: Is standardization required? True if Yes, False by default
    Returns: None
    """
    for participant in participants_subset:
        print('='*100)
        print(classifier)
        start_time = timeit.default_timer()
        model, metrics = run_model(model_dict=classifier_dictionary[classifier], model_data=model_data, participant_id=participant, scale=scale)
        elapsed = timeit.default_timer() - start_time
        if participant not in participant_dictionary.keys():
            participant_dictionary[participant] = {}
        participant_dictionary[participant][classifier] = {'model': model, 'metrics': metrics}
        participant_dictionary[participant][classifier]['metrics']['time (sec)'] = elapsed
        print('\n\n')
    return None


# # Description of the dataset - University Trial

# .csv files in this archive are in the following format:
# The first row is the initial time of the session expressed as unix timestamp in UTC.
# The second row is the sample rate expressed in Hz.
# 
# TEMP.csv
# Data from temperature sensor expressed degrees on the Celsius (°C) scale.
# 
# EDA.csv
# Data from the electrodermal activity sensor expressed as microsiemens (μS).
# 
# BVP.csv
# Data from photoplethysmograph (PPG).
# 
# ACC.csv
# Data from 3-axis accelerometer sensor. The accelerometer is configured to measure acceleration in the range [-2g, 2g]. Therefore the unit in this file is 1/64g.
# Data from x, y, and z axis are respectively in first, second, and third column.
# 
# IBI.csv
# Time between individuals heart beats extracted from the BVP signal.
# No sample rate is needed for this file.
# The first column is the time (respect to the initial time) of the detected inter-beat interval expressed in seconds (s).
# The second column is the duration in seconds (s) of the detected inter-beat interval (i.e., the distance in seconds from the previous beat).
# 
# HR.csv
# Average heart rate extracted from the BVP signal.The first row is the initial time of the session expressed as unix timestamp in UTC.
# The second row is the sample rate expressed in Hz.
# 
# 
# tags.csv
# Event mark times.
# Each row corresponds to a physical button press on the device; the same time as the status LED is first illuminated.
# The time is expressed as a unix timestamp in UTC and it is synchronized with initial time of the session indicated in the related data files from the corresponding session.

# # Colab
# from google.colab import drive
# drive.mount('/content/drive')
# directory = '/content/drive/MyDrive/Colab Notebooks/datasets/Stress-Predict-Dataset-main/Raw_data'

# ## Sensor -> Dataframe reference dictionary

# In[21]:


features = {'HR': {'has_refresh': True, 
                  'columns':{0: 'HR'},
                  },
            'EDA': {'has_refresh': True, 
                  'columns':{0: 'EDA'},
                  },
            'BVP': {'has_refresh': True, 
                  'columns':{0: 'BVP'},
                  },
            'IBI': {'has_refresh': False, 
                  'columns':{0: 'IBI_DETECT_TIME', 1: 'IBI_DURATION'},
                  },
            'TEMP': {'has_refresh': True, 
                  'columns':{0: 'TEMP'},
                  },
            'ACC': {'has_refresh': True, 
                  'columns':{0: 'x', 1: 'y', 2: 'z'},
                  },
            }


# ## ML Classifier reference dictionary

# In[22]:


classifier_dictionary = {
    "Random Forest": {'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), 
                      'params': {
                          "max_depth": range(5, 30, 5), 
                          "min_samples_leaf": range(1, 30, 2),
                          "n_estimators": range(100, 2000, 200)}},
    "Gradient Boost": {'model': GradientBoostingClassifier(random_state=RANDOM_STATE), 
                      'params': {
                          "learning_rate": [0.001, 0.01, 0.1], 
                          "n_estimators": range(1000, 3000, 200)}},
    "Linear SVM": {'model': SVC(random_state=RANDOM_STATE), 
                      'params': {
                          "kernel": ["rbf", "poly"], 
                          "gamma": ["auto", "scale"], 
                          "degree": range(1, 6, 1)}},
    "Nearest Neighbors": {'model': KNeighborsClassifier(n_jobs=-1), 
                      'params': {
                          'n_neighbors': [3, 5, 11, 19], 
                          'weights': ['uniform', 'distance'], 
                          'metric': ['euclidean', 'manhattan']}},
    "XGBoost": {'model': XGBClassifier(), 
                      'params': {
                        'min_child_weight': [1, 5, 10],
                        'gamma': [0.5, 1, 1.5, 2, 5],
                        'subsample': [0.6, 0.8, 1.0],
                        'colsample_bytree': [0.6, 0.8, 1.0],
                        'max_depth': [3, 4, 5], "n_estimators": [300, 600],
                        "learning_rate": [0.001, 0.01, 0.1]}}
}


# # Build merged dataset
# 
# Participant 1 has been excluded as the timestamps do not correspond to experment durations

# In[23]:


model_data = create_participant_dict()


# In[24]:


participants = np.fromiter(model_data.keys(), dtype=int)
participants


# # Split train and test participants

# In[25]:


train_split = round(TRAIN_PARTICIPANT_PROPORTION * len(participants))
#valid_split = round(train_split + 0.2 * len(participants))

participants_train = participants[:train_split]
#participants_valid = participants[train_split:valid_split]
participants_test = participants[train_split:]

len(participants_train), len(participants_test)
print({'Training Participants': participants_train,
       'Training Count': len(participants_train),
       'Test Participants': participants_test,
       'Test Count': len(participants_test),
       'Total Number of Participants:': len(participants_train)+len(participants_test)})


# ### Change n_participants from 35 to 24 to keep the test participants excluded during exploration

# In[26]:


n_participants = len(participants_train)
n_participants


# # Exploratory Data Analysis

# ## Check for consistent (numerical) data types across features

# In[50]:


get_participant_data(model_data, 2).dtypes


# **IBI_DURATION is not numeric. Will be converted**
# 
# **Time will be converted from UNIX timestamp to a readable datetime format**

# In[51]:


for participant in participants_train:
    model_data[participant]['data']['IBI_DURATION'] = model_data[participant]['data']['IBI_DURATION'].apply(pd.to_numeric)
    model_data[participant]['data']['time'] = pd.to_datetime(model_data[participant]['data']['time'], unit = "s", )
for participant in participants_test:
    model_data[participant]['data']['IBI_DURATION'] = model_data[participant]['data']['IBI_DURATION'].apply(pd.to_numeric)
    model_data[participant]['data']['time'] = pd.to_datetime(model_data[participant]['data']['time'], unit = "s", )

get_participant_data(model_data, 4).dtypes


# In[52]:


get_participant_data(model_data, 27).describe()


# In[53]:


get_participant_data(model_data, 2).describe()


# In[54]:


get_participant_data(model_data, 19).describe()


# ## Aggregate train participant data to make visualization easier

# In[55]:


data = get_participant_data(model_data, 2)
for participant in participants_train[1:]:
    data = pd.concat([data, get_participant_data(model_data, participant)], ignore_index=True)
    data.reset_index(drop=True, inplace=True)


# In[56]:


data


# ## Plot max, min and mean of TEMP

# In[57]:


plt.figure(figsize=(30,16), dpi=300)
min_data = data.loc[data.groupby(['participant']).TEMP.idxmin()]
max_data = data.loc[data.groupby(['participant']).TEMP.idxmax()]

plt.xlim(2, max(participants_train)+1)
plt.xticks(range(1, max(participants_train)+1))
plt.xlabel('Participant Number')
plt.ylabel('Recorded Temperature (°C)')

sns.lineplot(data=min_data, x='participant', y='TEMP', color='b', legend=False)
sns.lineplot(data=max_data, x='participant', y='TEMP', color='r', legend=False)

plt.axhline(data['TEMP'].mean(), linestyle="--", color='black')
plt.legend(title='Legend', loc='upper right', labels=['Lowest recorded value', 'Highest recorded value', 'Mean of recorded values'])


# ## Plot max, min and mean of BVP

# In[58]:


plt.figure(figsize=(30,16), dpi=300)
min_data = data.loc[data.groupby(['participant']).BVP.idxmin()]
max_data = data.loc[data.groupby(['participant']).BVP.idxmax()]

plt.xlim(2, max(participants_train)+1)
plt.xticks(range(1, max(participants_train)+1))
plt.xlabel('Participant Number')
plt.ylabel('Recorded BVP')

sns.lineplot(data=min_data, x='participant', y='BVP', color='b')
sns.lineplot(data=max_data, x='participant', y='BVP', color='r')

plt.axhline(data['BVP'].mean(), linestyle="--", color='black')
plt.legend(title='Legend', loc='upper right', labels=['Lowest recorded value', 'Highest recorded value', 'Mean of recorded values'])


# ## Investigate TEMP

# **Sharp drop in temperature at the end of the experiment occurs for most participants but not all. This is probably due to the participants taking off the wearable prior to ending the experiment.**

# In[59]:


plot_participant_progression(data, 19, 'TEMP')


# In[60]:


plot_participant_progression(data, 20, 'TEMP')


# In[61]:


plot_participant_progression(data, 12, 'TEMP')


# In[62]:


plot_participant_progression(data, 5, 'TEMP')


# ## Plot max, min and mean of HR

# In[63]:


plt.figure(figsize=(30,16), dpi=300)
min_data = data.loc[data.groupby(['participant']).HR.idxmin()]
max_data = data.loc[data.groupby(['participant']).HR.idxmax()]

plt.xlim(2, max(participants_train)+1)
plt.xticks(range(1, max(participants_train)+1))

plt.xlabel('Participant Number')
plt.ylabel('Heartrate (bpm)')
sns.lineplot(data=min_data, x='participant', y='HR', color='b')
sns.lineplot(data=max_data, x='participant', y='HR', color='r')

plt.axhline(data['HR'].mean(), linestyle="--", color='black')
plt.legend(title='Legend', loc='upper right', labels=['Lowest recorded value', 'Highest recorded value', 'Mean of recorded values', 'Mean of recorded values'])


# In[64]:


plot_participant_progression(data, 2, 'TEMP')


# In[65]:


plot_participant_progression(data, 2, 'HR')


# **The second stress test (Trier Social Scale Test) appears to have a significant impact on the heart rate of participant 2. Will require further investigation**

# In[66]:


plot_participant_progression(data, 12, 'HR')


# **Participant 12, does not appear to have a similar rise in heart rate**

# In[67]:


plot_participant_progression(data, 20, 'HR')


# **Participant 20 also appears to expreience a sharp rise in heart rate duuing the TSST similar to participant 12**

# In[68]:


plot_participant_progression(data, 26, 'HR')


# **Participant 26 however, does not appear to have a similar rise in heart rate**

# In[69]:


plot_participant_progression(data, 2, 'EDA')


# In[70]:


plot_participant_progression(data, 2, 'BVP')


# In[71]:


plot_participant_progression(data, 2, 'IBI_DURATION')


# ## Fill NaN values across participant features - Only for train participants

# In[72]:


nan_barplot(get_participant_data(model_data, 4))
get_participant_data(model_data, 4).isnull().sum()


# **Use ffill and bfill to fill in missing values across the dataset. Imputation is not preferred as it might alter the existing patterns in the dataset by 'learning' from other features. Filling with previous/next values are more appropriate as the expected sensor readings should not differ too much from the previous/next values.**

# In[73]:


for participant in participants_train:
    model_data[participant]['data'].fillna(method='ffill', inplace=True)
    model_data[participant]['data'].fillna(method='bfill', inplace=True)

nan_barplot(get_participant_data(model_data, 4))
get_participant_data(model_data, 4).isnull().sum()


# ## Correlation Heatmap | Participants picked at random
# 
# **Only indicates linear relationship**

# In[74]:


plt.figure(figsize=(30,16), dpi=300)
sns.heatmap(get_participant_data(model_data, 4).corr(), annot=True)


# In[75]:


plot_participant_progression(data, 4, 'HR')


# In[76]:


plot_participant_progression(data, 4, 'EDA')


# In[77]:


plot_participant_progression(data, 4, 'y')


# In[78]:


plot_participant_progression(data, 4, 'IBI_DURATION')


# **Participant 4 shows a notable change in EDA, HR and y-axis (ACC) readings when under stress**

# In[79]:


plt.figure(figsize=(30,16), dpi=300)
sns.heatmap(get_participant_data(model_data, 2).corr(), annot=True)


# In[80]:


plot_participant_progression(data, 2, 'HR')


# In[81]:


plot_participant_progression(data, 2, 'z')


# **Participant 2 shows a slight change in HR and z-axis (ACC) readings when under stress**

# In[82]:


plt.figure(figsize=(30,16), dpi=300)
sns.heatmap(get_participant_data(model_data, 22).corr(), annot=True)


# In[83]:


plot_participant_progression(data, 22, 'EDA')


# In[84]:


plot_participant_progression(data, 22, 'HR')


# In[85]:


plot_participant_progression(data, 22, 'TEMP')


# **Participant 22 shows a notable change in EDA, HR and TEMP readings when under stress**

# In[86]:


plt.figure(figsize=(30,16), dpi=300)
sns.heatmap(get_participant_data(model_data, 14).corr(), annot=True)


# In[87]:


plot_participant_progression(data, 14, 'EDA')


# In[88]:


plot_participant_progression(data, 14, 'HR')


# In[89]:


plot_participant_progression(data, 14, 'TEMP')


# **Participant 14 shows a notable change in EDA, HR and TEMP readings when under stress**

# # NOTE:
# 
# **The above plots confirm that different participants have different responses to stress and at varying intensities. On average, participants respond with a rise in temperature, electrodermal activity (increased perspiration), and z-axis (forward and backward movements). However, this is not always the case.**
# 
# **As a result, every participant will need to have a personalized model trained to their individual stress responses and all features except BVP will have to be retained.**
# 
# **BVP has consistently shown to have no impact in indicating if a person is under stress or not.**
# 

# pd.options.display.max_rows = 999
# train, test = slice_train_test(model_data[3]['data'], 0.2, np.inf)

# # ML Models
# 
# Priority given to recall score - false negatives are more detrimental

# # Gradient Boosting Classifier

# In[68]:


run_participants(participants, 'Gradient Boost', scale=False)


# # KNN Classifier

# In[32]:


run_participants(participants, 'Nearest Neighbors', scale=True)


# # Support Vector Classifier

# In[70]:


run_participants(participants, 'Linear SVM', scale=True)


# # XGBoost Classifier

# In[71]:


run_participants(participants, 'XGBoost', scale=False)


# # Random Forest Classifier

# In[72]:


run_participants(participants, 'Random Forest', scale=False)


# # Model Performance

# In[354]:


participant_dictionary


# In[74]:


for participant in participants:
    plot_model_performance(participant, 'model')


# In[75]:


plot_model_performance(2, 'time')


# In[76]:


plot_model_performance(4, 'time')


# In[77]:


plot_model_performance(22, 'time')


# In[78]:


plot_model_performance(2, 'model')


# In[79]:


plot_model_performance(4, 'model')


# In[80]:


plot_model_performance(22, 'model')


# # Experimantal! - Predictive Monitoring

# ## Set the probability threshold for each level of stress

# In[35]:


# For reference (RUN THIS CELL)
stress_thresholds = {
    0.0: 'NOT STRESSED',
    0.25: 'LIGHTLY STRESSED',
    0.50: 'MODERATELY STRESSED',
    0.70: 'WARNING',
    0.85: 'ALERT - STRESSED'
}


# In[38]:


def simulate_device(participant_id, classifier_name, participant_dictionary, stress_thresholds, scale=False):
    # Prepare test data. 
    dropped_features = ['label', 'time', 'participant', 'BVP']
    train, test, valid = slice_train_test1(model_data[participant_id]['data'], 
                                    buffer_size=5, 
                                    test_proportion=SLICE_TEST_PROPORTION, 
                                    window_size=WINDOW_SIZE)
    X_train = train.drop(dropped_features, axis=1)
    X_test = test.drop(dropped_features, axis=1)
    y_train = train['label']
    y_test = test['label']
    
    test_slice_size = int(WINDOW_SIZE * SLICE_TEST_PROPORTION)
    
    if scale == True:
        print('Scaling data...')
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    trained_model = participant_dictionary[participant_id][classifier_name]['model']
    #y_probs = trained_model.predict_proba(X_test)
    
    y_probs_positive = list()
    
    #y_probs = list()
    print('Simulating Device Alerts...')
    print('Testing in', test_slice_size, 'second windows')
    window_count = 1
    window_positive_probability = list()
    all_average_window_probability = list()
    for i in range(0, len(X_test), test_slice_size):
        print('\nTesting window', window_count)
        test_slice = X_test[i: i+test_slice_size]
        for row in test_slice:
            # Make predictions one second at a time and store results
            prediction = trained_model.predict_proba([row])
            positive_probability = prediction[:, 1][0]
            y_probs_positive.append(positive_probability)
            window_positive_probability.append(positive_probability)
            if positive_probability < 0.25:
                print('NOT STRESSED :',positive_probability)
            elif positive_probability > 0.25 and positive_probability < 0.50:
                print('LIGHTLY STRESSED:', positive_probability)
            elif positive_probability > 0.50 and positive_probability < 0.70:
                print('MODERATELY STRESSED:', positive_probability)
            elif positive_probability > 0.70 and positive_probability < 0.85:
                print('WARNING:', positive_probability)
            else:
                print('ALERT - STRESSED:', positive_probability)
        window_average_probability = np.mean(window_positive_probability)
        print('Window average positive probability:', window_average_probability)
        if window_average_probability < 0.25:
            print('WINDOW RESULT: NOT STRESSED')
        elif window_average_probability > 0.25 and window_average_probability < 0.50:
            print('WINDOW RESULT: LIGHTLY STRESSED')
        elif window_average_probability > 0.50 and window_average_probability < 0.70:
            print('WINDOW RESULT: MODERATELY STRESSED')
        elif window_average_probability > 0.70 and window_average_probability < 0.85:
            print('WINDOW RESULT: WARNING')
        else:
            print('WINDOW RESULT: ALERT - STRESSED')
        all_average_window_probability.append(window_average_probability)
        window_positive_probability = list()
        window_count += 1
        
    # Plot stress probability with colour coded result in 35 (window_size) second chunks
    fig, ax = plt.subplots(figsize=(14, 6))
    graph = sns.lineplot(data=y_probs_positive)
    graph.axvline(i, color='g', alpha=0.2, lw=3, label='NOT STRESSED')
    graph.axvline(i, color='b', alpha=0.2, lw=3, label='LIGHTLY STRESSED')
    graph.axvline(i, color='y', alpha=0.2, lw=3, label='MODERATELY STRESSED')
    graph.axvline(i, color='m', alpha=0.2, lw=3, label='WARNING')
    graph.axvline(i, color='r', alpha=0.2, lw=3, label='ALERT - STRESSED')
    for i in range(test_slice_size, len(y_probs_positive), test_slice_size):
        current = all_average_window_probability[i//test_slice_size]
        if current < 0.25:
            graph.axvline(i, color='g', alpha=0.2, lw=test_slice_size)
        elif current > 0.25 and current < 0.50:
            graph.axvline(i, color='b', alpha=0.2, lw=test_slice_size)
        elif current > 0.50 and current < 0.70:
            graph.axvline(i, color='y', alpha=0.2, lw=test_slice_size)
        elif current > 0.70 and current < 0.85:
            graph.axvline(i, color='m', alpha=0.2, lw=test_slice_size)
        else:
            graph.axvline(i, color='r', alpha=0.2, lw=test_slice_size)
    plt.legend(loc = 2, bbox_to_anchor = (1,1))
    plt.ylabel('Probability of being stressed')
    plt.xlabel('Seconds')
    #plt.legend(labels=["NOT STRESSED","LIGHTLY STRESSED", "MODERATELY STRESSED", "WARNING", "ALERT - STRESSED"], loc = 2, bbox_to_anchor = (1,1))
    return None


# In[39]:


simulate_device(4, 'Nearest Neighbors', participant_dictionary, stress_thresholds, scale=True)


# # END
