import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold



def load_data():
    dataset = pd.read_csv('D:/MULTIMEDIA/MACHINE_LEARNING_THAY_QUANG/LUAN AN TIEN SI/ImAdaBoost_KFold/ProcessingData/dataset/transfusion.csv')
    dataset_desc = dataset.describe(include = 'all')
    transfusion_map = {1:1, 0:-1}
    dataset['whether he/she donated blood in March 2007'] = dataset['whether he/she donated blood in March 2007'].map(transfusion_map)
    X = dataset.drop(['whether he/she donated blood in March 2007'], axis = 1)
    y = dataset['whether he/she donated blood in March 2007']
    X = np.array(X)
    y = np.array(y)
    return X,y

