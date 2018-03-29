from sklearn import preprocessing

def normalize_feature(X):
    return preprocessing.scale(X)