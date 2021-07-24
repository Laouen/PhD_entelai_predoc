from sklearn import datasets

def preprocess_data():
    data = datasets.load_wine(as_frame=True)

    return {
        'X': data['data'],
        'y': data['target']
    }