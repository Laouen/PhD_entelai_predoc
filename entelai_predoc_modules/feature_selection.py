from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def feature_selection(X_train, y_train, X_test):

    ###initialize Boruta
    forest = RandomForestRegressor(
        n_jobs = -1, 
        max_depth = 5
    )
    boruta = BorutaPy(
        estimator = forest, 
        n_estimators = 'auto',
        max_iter = 100 # number of trials to perform
    )

    ### fit Boruta (it accepts np.array, not pd.DataFrame)
    boruta.fit(X_train, y_train)

    ### Selected features 
    selected_features = boruta.support_ | boruta.support_weak_

    print(f'Total feature selected: {np.sum(selected_features)}')
    
    return {
        'X_train': X_train[:, selected_features],
        'X_test': X_test[:, selected_features],
        'selected_features': selected_features
    }
