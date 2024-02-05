import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_data():
    dataset = pd.read_csv('D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/LUAN AN TIEN SI/ImAdaBoost_KFold/ProcessingData/dataset/yeast.csv')
    dataset_desc = dataset.describe(include='all')
    yeast_map = {'ME2': 1, 'CYT': -1, 'ERL': -1, 'EXC': -1, 'ME1': -1, 'ME3': -1, 'MIT': -1, 'NUC': -1, 'POX': -1, 'VAC': -1}
    dataset['name'] = dataset['name'].map(yeast_map)
    X = dataset.drop(['name'], axis=1)
    y = dataset['name']
    X = np.array(X)
    y = np.array(y)
    return X, y






