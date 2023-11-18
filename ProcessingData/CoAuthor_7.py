import pandas as pd

def load_data():
    dataset = pd.read_csv('./ProcessingData/dataset/ACOMP_13_G2_0.06_75_1250.csv')
    X = dataset.values[:, 0:-1].astype(float)
    y = dataset.values[:, 7].astype(float)
    return X, y


