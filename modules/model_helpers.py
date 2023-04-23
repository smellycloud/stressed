from modules import constants
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    #train, test = slice_train_test(participant_data, constants.SLICE_TEST_PROPORTION, SLICE_CHUNK_SIZE)
    train, test, valid = slice_train_test1(participant_data, buffer_size=5, test_proportion=constants.SLICE_TEST_PROPORTION, WINDOW_SIZE=constants.WINDOW_SIZE)
    X_train = train.drop(dropped_features, axis=1)
    X_test = test.drop(dropped_features, axis=1)
    y_train = train['label']
    y_test = test['label']

    if scale == True:
        print('Scaling data...')
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    rscv = RandomizedSearchCV(model, scoring='accuracy', cv=constants.CV_FOLDS,
                              param_distributions=params, refit=True,
                              n_jobs=-1, random_state=constants.RANDOM_STATE,
                              return_train_score=True, verbose=constants.RSCV_VERBOSITY)
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
