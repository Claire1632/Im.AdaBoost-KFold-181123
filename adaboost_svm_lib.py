from numpy.core.records import array
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
# from data import Vertebral_column
# from data import Co_Author
# from data import indian_liver_patient
# #from data import spect_heart
# from src.wsvm import trainning_of_adaboost as toa
#from sklearn.metrics import f1_score
from sklearn.metrics  import classification_report,precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing
from sklearn import metrics
import math
from sklearn.svm import SVC
# from data import indian_liver_patient
from collections import Counter

# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
    confusion_matrix(y_test, y_pred))
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
    print("Report : ",
    classification_report(y_test, y_pred))

def is_tomek(X,y, class_type):
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X)
    nn_index = nn.kneighbors(X, return_distance=False)[:, 1]
    links = np.zeros(len(y), dtype=bool)
    # find which class to not consider
    class_excluded = [c for c in np.unique(y) if c not in class_type]
    X_dangxet = []
    X_tl = []
    # there is a Tomek link between two samples if they are both nearest
    # neighbors of each others.
    for index_sample, target_sample in enumerate(y):
        if target_sample in class_excluded:
            continue
        if y[nn_index[index_sample]] != target_sample:
            if nn_index[nn_index[index_sample]] == index_sample:
                X_tl.append(index_sample)
                X_dangxet.append(nn_index[index_sample])
                links[index_sample] = True

    return links,X_dangxet,X_tl

def Gmean(y_test,y_pred):
    cm_WSVM = metrics.confusion_matrix(y_test, y_pred)
    sensitivity = cm_WSVM[1,1]/(cm_WSVM[1,0]+cm_WSVM[1,1])
    specificity = cm_WSVM[0,0]/(cm_WSVM[0,0]+cm_WSVM[0,1])
    gmean = math.sqrt(sensitivity*specificity)
    return gmean

def data_tomelinks(model,X_train,y_train,X_test,y_test,n_neighbors):
    links,xdx,xtl = is_tomek(X_train,y_train,class_type=[-1.0])
    model.fit(X_train,y_train)
    y_predict = model.predict(X_test)
    gmean = Gmean(y_test,y_predict)
    nn2 = NearestNeighbors(n_neighbors=n_neighbors)
    nn2.fit(X_train)
    y_nn = []
    y_check_pos = np.array(y_train <-2)
    for ind,i in enumerate(xdx):                                # xdx: cac nhan +1 dang xet'
        y_pred = float(model.predict([X_train[i]]))       #
        if y_pred == -1.0:                          
            knn_X = (nn2.kneighbors([X_train[i]])[1]).tolist() 
            for j in knn_X[0]:
                y_nn.append(y_train[j])    # gom nhãn láng giềng của X_train[i] bị dự đoán sai vào y_nn
        else:
            # knn_X = (nn2.kneighbors([X_train[i]])[1]).tolist()  
            # for j in knn_X[0]:
            #     if y_train[j] == -1.0:   
            #         y_check_pos[j] = True   # xóa nhãn âm xung quanh nhãn dương đc phân loại đúng
            y_check_pos[xtl[ind]] = True

    y_nn = np.array(y_nn)
    if len(y_nn)>0:
        y_nn = np.array_split(y_nn, len(y_nn)/n_neighbors)
    y_check_neg = np.array(y_train <-2)
    for i in range(0,len(y_nn)):      #
        if 1 not in y_nn[i][1:]:      # Nếu không có nhãn 1 xung quanh X_train[i] bị dự đoán sai => xóa X_train[i]
            y_check_neg[xdx[i]] = True

    y_check = y_check_neg | y_check_pos
    
    sample_indices_ = np.flatnonzero(np.logical_not(y_check)) 
    ytl = _safe_indexing(y_train, sample_indices_)
    Xtl = _safe_indexing(X_train, sample_indices_)
    return Xtl,ytl,y_predict,gmean

def lfb(model,T,X_train,y_train,X_test,y_test,n_neighbors,thamso1,thamso2): #loop find the best
    gmean = 0
    for i in range(0,T):
        X_train, y_train, y_predict, gmean2 = data_tomelinks(model,X_train,y_train,X_test,y_test,n_neighbors)
        metr(X_train,gmean2,y_test,y_predict)
        if ((gmean2 - gmean) <= thamso1) or (gmean2 > thamso2):
            print("_____Gmean_ERROR!!!____")
            return X_train, y_train
        else:
            gmean = gmean2
    return X_train, y_train

def metr(X_train,gmean,y_test,test_pred):
    print("So luong samples: ",len(X_train))
    print("G mean: ",gmean)
    print("Accuracy: ",accuracy_score(y_test,test_pred))
    print("Ma tran nham lan: \n",confusion_matrix(y_test, test_pred))

# Driver code
def main():
    # X_train, y_train, X_test, y_test = indian_liver_patient.load_data(test_size=0.3)
    # print(len(y_train))
    svc=SVC(probability=True, kernel='linear')
    abc =AdaBoostClassifier(n_estimators=50, learning_rate=1)
    # model2 = svc.fit(X_train, y_train)   
    # test_pred = model2.predict(X_test)
    # print(classification_report(y_test, test_pred))
    # metr(X_train,Gmean(y_test,test_pred),y_test,test_pred)
    X_train, y_train, X_test, y_test = Co_Author.load_data(test_size=0.3)
    print(len(X_train))
    print('Original dataset shape %s' % Counter(y_train))
    print("ADA Boost with W.SVM starting...\n")
    thamso1 = 0
    thamso2 = 1
    T = 4
    n_neighbors = 5

    X_train_new, y_train_new = lfb(svc,T,X_train,y_train,X_test,y_test,n_neighbors,thamso1,thamso2)
    print('NEWWWWWWW dataset shape %s' % Counter(y_train_new))
    model = svc.fit(X_train_new, y_train_new)    
    test_pred = model.predict(X_test)
    print(classification_report(y_test, test_pred))
    metr(X_train_new,Gmean(y_test,test_pred),y_test,test_pred)

if __name__=="__main__":
    main()