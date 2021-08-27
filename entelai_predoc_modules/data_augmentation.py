from imblearn.over_sampling import BorderlineSMOTE
import numpy as np

def data_augmentation(X, y, multiple):

    classes, counts = np.unique(y, return_counts=True)
    
    if multiple == 'balanced':
        resample_dict = {
            c: counts.max() for c in classes
        }
    else:
        resample_dict = {
            classes_: count
            for classes_, count in zip(classes,counts)
        }

    smote = BorderlineSMOTE(sampling_strategy=resample_dict)
    resampled_X, resampled_y = smote.fit_resample(X, y)

    return {
        'X': resampled_X,
        'y': resampled_y
    }
