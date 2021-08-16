from imblearn.over_sampling import BorderlineSMOTE
import numpy as np

def data_augmentation(X, y, multiple):

    # Note: Se asume que las clases existentes son 0, 1.
    more_frequent_class = np.bincount(y).argmax()
    less_frequent_class = abs(more_frequent_class - 1)

    if multiple == 'balanced':
        total_features = len(np.where(y == more_frequent_class)[0])
        resample_dict = {
            more_frequent_class: total_features,
            less_frequent_class: total_features
        }
    else:
        resample_dict = {
            more_frequent_class: len(np.where(y == more_frequent_class)[0]),
            less_frequent_class: len(np.where(y == less_frequent_class)[0]) * multiple
        }

    smote = BorderlineSMOTE(sampling_strategy=resample_dict)
    resampled_X, resampled_y = smote.fit_resample(X, y)

    return {
        'X': resampled_X,
        'y': resampled_y
    }
