from modules import constants
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, accuracy_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

import timeit
import itertools
from datetime import datetime
def get_participant_data(model_data, participant_id):
    """
    Args:
    model_data: Dictionary with all participant dataframes (dict)
    participant_id: Participant ID (int)

    Returns:
    A dataframe corresponding to an individual participant
    """
    return model_data[participant_id]['data']

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
    df = pd.read_csv(constants.directory+'/S'+str(participant_number).zfill(2)+'/'+key+'.csv', header=None)
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
    tags = pd.read_csv(constants.directory+'/S'+str(participant_number).zfill(2)+'/tags_S'+str(participant_number).zfill(2)+'.csv', header=None)
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

def build_dataframe(participant_number):
    '''
    Merge all participant features into a single dataframe
    Args:
    data: Participant sensor data
    Returns:
    A dataframe with all sensor data merged by timestamp
    '''
    if participant_number < 1 or participant_number > constants.n_participants:
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

def create_participant_dict():
    """
    Returns:
    A dictionary with all participant dataframes
    """
    participant_data = dict()
    for i in range(2, constants.n_participants+1):
        extracted_data = append_labels(build_dataframe(i), i)
        if extracted_data is not None:
            extracted_data.reset_index(drop=True, inplace=True)
            participant_data[i] = {'data': None}
            participant_data[i]['data'] = extracted_data
    return participant_data

def slice_train_test1(data, buffer_size=2, drop_buffer=True, test_proportion=constants.SLICE_TEST_PROPORTION, WINDOW_SIZE=constants.WINDOW_SIZE):
    """
    Continuously grabs 20 second and 80 second snapshots for every 100 seconds and splits them at random
    into test and train dataframes respectively. Order is maintained.

    Args:
    data: Participant dataframe
    test_proportion: Proportion of data to use for testing (default: 0.2)
    constants.WINDOW_SIZE: Size of chunks to split into train and test (default: 100).
                If unspecified/out of range, will automatically split into 10 equal chunks.
    Returns:
    train, test, valid : A tuple with train and test data
    """
    np.random.seed(constants.RANDOM_STATE)
    if data is None:
        print('No data provided')
        return None, None
    if test_proportion >= 1 or test_proportion <= 0:
        print('Invalid test proportion')
        return None, None

    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    data.reset_index(drop=True, inplace=True)

    n_rows = data.shape[0]
    print('Total number of rows:', n_rows)
    train = pd.DataFrame()
    test = pd.DataFrame()
    buffer = pd.DataFrame()

    if constants.WINDOW_SIZE > n_rows or constants.WINDOW_SIZE < (buffer_size*2)+1:
        constants.WINDOW_SIZE = int(n_rows // 10)
        print('Invalid window size. Setting window size to', constants.WINDOW_SIZE)

    print('Window size:', constants.WINDOW_SIZE)
    test_slice_size = int(test_proportion * constants.WINDOW_SIZE)
    print('Test slice size:', test_slice_size)

    count = 0
    start_index = 0

    for index in range(1, n_rows, constants.WINDOW_SIZE):
        if n_rows - index < constants.WINDOW_SIZE:
            break
        end = index + constants.WINDOW_SIZE
        data_slice = data.iloc[index:end]
        data_slice.reset_index(drop=True, inplace=True)
        while True:
                picked_top = int(np.random.uniform(buffer_size, len(data_slice) - buffer_size))
                #picked_top = int(np.random.uniform(0, len(data_slice)))
                picked_bottom = picked_top + test_slice_size
                buffer_top = picked_top - buffer_size
                buffer_bottom = picked_bottom + buffer_size
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
            train = pd.concat([train, data_slice[0:buffer_top]], ignore_index=True)
            buffer_top_data = data_slice[buffer_top:picked_top]
            buffer = pd.concat([buffer, buffer_top_data], ignore_index=True)

            picked_test = data_slice[picked_top:picked_bottom]
            test = pd.concat([test, picked_test], ignore_index=True)

            buffer_bottom_data = data_slice[picked_bottom:buffer_bottom]
            buffer = pd.concat([buffer, buffer_bottom_data], ignore_index=True)

            train = pd.concat([train, data_slice[buffer_bottom:index+constants.WINDOW_SIZE]], ignore_index=True)
            start_index = index
        else:
            print('TODO')
            return None, None

    n_leftovers = n_rows - (len(buffer) + len(train) + len(test))
    if n_leftovers > 0:
        print('Leftover rows:', n_leftovers, '\nMoving into train set...\n')
        leftovers = data[n_rows-n_leftovers:]
        train = pd.concat([train, leftovers], ignore_index=True)

    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    buffer.reset_index(drop=True, inplace=True)
    print(len(train), len(buffer), len(test))
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


def slice_train_test(data, test_proportion=constants.SLICE_TEST_PROPORTION, chunk_size=constants.WINDOW_SIZE):
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
