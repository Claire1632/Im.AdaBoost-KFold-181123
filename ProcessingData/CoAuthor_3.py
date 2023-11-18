import pandas as pd

def load_data():
    dataset = pd.read_csv('./ProcessingData/dataset/ACOMP_22_G3_0.04_33_825.csv')
    X = dataset.values[:, 0:-1].astype(float)
    y = dataset.values[:, 7].astype(float)
    return X, y


