import pandas as pd

def load_data():
    dataset = pd.read_csv('./ProcessingData/dataset/ACOMP_12_G2_0.08_70_875.csv')
    X = dataset.values[:, 0:-1].astype(float)
    y = dataset.values[:, 7].astype(float)
    return X, y


