import pandas as pd

def load_data():
    dataset = pd.read_csv('./ProcessingData/dataset/ACOMP_16_G2_0.0161_63_3913.csv')
    X = dataset.values[:, 0:-1].astype(float)
    y = dataset.values[:, 7].astype(float)
    return X, y


