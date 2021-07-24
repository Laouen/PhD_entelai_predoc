def feature_extraction(X, columns_to_include=[], columns_to_drop=[]):

    if len(columns_to_include) > 0:
        res = X[columns_to_include]
    else:
        res = X[[c for c in X.columns if c not in columns_to_drop]]
    
    return {'X': res}
