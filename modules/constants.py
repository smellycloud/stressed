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
