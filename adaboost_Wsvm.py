import numpy as np
# from data import Vertebral_column
# from data import Co_Author
# from data import indian_liver_patient
#from data import spect_heart
from wsvm.application import Wsvm
from svm.application import Svm
from sklearn.svm import SVC
#from sklearn.metrics import f1_score
from sklearn.metrics  import classification_report,precision_recall_fscore_support as score
from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score,f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing
from sklearn import metrics
import math
from datetime import datetime

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
    cm_WSVM = confusion_matrix(y_test, y_pred)
    sensitivity = cm_WSVM[1,1]/(cm_WSVM[1,0]+cm_WSVM[1,1])
    specificity = cm_WSVM[0,0]/(cm_WSVM[0,0]+cm_WSVM[0,1])
    gmean = math.sqrt(sensitivity*specificity)
    return sensitivity,specificity,gmean

def data_tomelinks(X_train,y_train,X_test,y_test,n_neighbors,clf=None):
    links,xdx,xtl = is_tomek(X_train,y_train,class_type=[-1.0])
    print(type(X_train))
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_train)
    sensitivity,specificity,gmean = Gmean(y_train,y_predict)
    nn2 = NearestNeighbors(n_neighbors=n_neighbors)
    nn2.fit(X_train)
    y_nn = []
    y_check_pos = np.array(y_train <-2)
    for ind,i in enumerate(xdx):
        y_pred = clf.predict([X_train[i]])  #
        if y_pred == -1.0:                          
            knn_X = (nn2.kneighbors([X_train[i]])[1]).tolist() 
            for j in knn_X[0]:
                y_nn.append(y_train[j])    # gom nhãn láng giềng của X_train[i] bị dự đoán sai vào y_nn
        else:
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

def lfb(f,T,X_train,y_train,X_test,y_test,n_neighbors,clf,thamso1,thamso2): #loop find the best
    gmean = 0
    for i in range(0,T):
        X_train, y_train, y_predict, gmean2 = data_tomelinks(X_train,y_train,X_test,y_test,n_neighbors,clf)
        clf.fit(X_train, y_train)
        pred2 = clf.predict(X_train)
        sp,se,gmean = Gmean(y_train,pred2)
        #metr(X_train,y_train,pred2,sp,se,gmean)
        #metr_text(f,X_train,y_train,pred2,sp,se,gmean)
        if ((gmean2 - gmean) <= thamso1) or (gmean2 > thamso2):
            #f.write("\n_____Gmean_ERROR!!!____\n")
            #print("\n_____Gmean_ERROR!!!____\n")
            return X_train, y_train
        else:
            gmean = gmean2
    return X_train, y_train
def metr_text(f,X_train,y_test,test_pred,se,sp,gmean):
    #se,sp,gmean = Gmean(y_test,test_pred)
    f.write(f"\n\nSo luong samples: {len(X_train)}\n")
    f.write("\n"+str(classification_report(y_test, test_pred)))
    f.write(f"\nSP      : {sp:0.4f}")
    f.write(f"\nSE      : {se:0.4f}")
    f.write(f"\nGmean   : {gmean:0.4f}")
    f.write(f"\nF1 Score: {f1_score(y_test, test_pred):0.4f}")
    f.write(f"\nAccuracy: {accuracy_score(y_test,test_pred):0.4f}")
    f.write(f"\nAUC     : {roc_auc_score(y_test, test_pred):0.4f}")
    f.write("\n\nMa tran nham lan: \n"+str(confusion_matrix(y_test, test_pred)))

def metr(X_train,y_test,test_pred,se,sp,gmean):
    #se,sp,gmean = Gmean(y_test,test_pred)
    print("So luong samples: ",len(X_train))
    print("\n",classification_report(y_test, test_pred))
    print("SP      : ",sp)
    print("SE      : ",se)
    print("Gmean   : ",gmean)
    print("F1 Score: ",f1_score(y_test, test_pred))
    print("Accuracy: ",accuracy_score(y_test,test_pred))
    print("AUC     : ",roc_auc_score(y_test, test_pred))
    print("Ma tran nham lan: \n",confusion_matrix(y_test, test_pred))

def svm(X_train, y_train,X_test):
    model = Svm(C)
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    return test_pred

def svm_lib(X_train, y_train,X_test):
    svc=SVC(probability=True, kernel='linear')
    model = svc.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def wsvm(X_train, y_train,X_test):
    model = Wsvm(C)
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    return test_pred

C=100
thamso1 = 0
thamso2 = 1
T = 4
n_neighbors = 5
test_size = [0.2,0.3,0.4]
data = [Vertebral_column, indian_liver_patient, Co_Author]
time = datetime.now().strftime("%d%m%Y_%H%M%S")
filepath = "D:/Fuzzy/fuzzy_svm/text_script"

#cac bo phan lop dung: #None = W.svm

#svc lib
svc = SVC(probability=True, kernel='linear')
#svm scratch
svm_scr = Svm(C)
#W.svm
wsvm_scr = Wsvm(C)
#fuzzy
    #---

# f = open("D:/Fuzzy/fuzzy_svm/file1101.txt", "a")
# f.write(f"\nM= {M}, C={C}, thamso1 = {thamso1}, thamso2={thamso2},T={T}, n_neighbors= {n_neighbors}  \n")
for dataset in data:
    filename = (str(dataset).split("\\")[-1]).split(".")[0]
    f = open(f"{filepath}/Data_{filename}_{time}.txt", "w")
    f.write(f"\nC = {C}, thamso1 = {thamso1}, thamso2 = {thamso2}, T = {T}, n_neighbors = {n_neighbors}  \n")
    f.write(f"\n\n\tUSING DATASET : {filename}\n")
    print(f"\n\tUSING DATASET : {filename}\n")
    for testsize in test_size:
        X_train, y_train, X_test, y_test = dataset.load_data(test_size=testsize)

        print(f"\t======== TestSize: {testsize} ========")
        f.write(f"\n\n\t======== TestSize: {testsize}========\n\n")
        print("So luong sample ban dau: ",len(X_train))
        f.write("\n\t====== NOT USING TOMEKLINKS ========== \n")
        #Svm library
        f.write("\n\nSVM LIBRARY starting...\n")
        print("SVM LIBRARY starting...\n")
        test_pred = svm_lib(X_train, y_train,X_test)
        sp,se,gmean = Gmean(y_test,test_pred)
        metr(X_train,y_test,test_pred,sp,se,gmean)
        metr_text(f,X_train,y_test,test_pred,sp,se,gmean)

        #Svm scratch
        f.write("\n\nSVM starting...\n")
        print("SVM starting...\n")
        test_pred = svm(X_train, y_train,X_test)
        sp,se,gmean = Gmean(y_test,test_pred)
        metr(X_train,y_test,test_pred,sp,se,gmean)
        metr_text(f,X_train,y_test,test_pred,sp,se,gmean)

        #Wsvm
        f.write("\n\nW.SVM starting...\n")
        print("W.SVM starting...\n")
        test_pred = wsvm(X_train, y_train,X_test)
        sp,se,gmean = Gmean(y_test,test_pred)
        metr(X_train,y_test,test_pred,sp,se,gmean)
        metr_text(f,X_train,y_test,test_pred,sp,se,gmean)
        
        f.write("\n\n\t========== USING TOMEKLINKS ==========\n")
        #Svm library
        f.write("\n\nSVM LIBRARY starting...\n")
        print("SVM LIBRARY starting...\n")
        X_train_new, y_train_new = lfb(f,T,X_train,y_train,X_test,y_test,n_neighbors,svc, thamso1,thamso2) 
        test_pred_tl = svm_lib(X_train_new, y_train_new,X_test)
        sp,se,gmean = Gmean(y_test,test_pred_tl)
        metr(X_train,y_test,test_pred_tl,sp,se,gmean)
        metr_text(f,X_train,y_test,test_pred_tl,sp,se,gmean)

        #Svm
        f.write("\n\nSVM starting...\n")
        print("SVM  starting...\n")
        X_train_new, y_train_new = lfb(f,T,X_train,y_train,X_test,y_test,n_neighbors,svm_scr, thamso1,thamso2) 
        test_pred_tl = svm(X_train_new, y_train_new,X_test)
        sp,se,gmean = Gmean(y_test,test_pred_tl)
        metr(X_train,y_test,test_pred_tl,sp,se,gmean)
        metr_text(f,X_train,y_test,test_pred_tl,sp,se,gmean)

        ##ada.Wsvm
        f.write("\n\nADA Boost with W.SVM starting...\n")
        print("ADA Boost with W.SVM starting...\n")
        X_train_new, y_train_new = lfb(f,T,X_train,y_train,X_test,y_test,n_neighbors,wsvm_scr,thamso1,thamso2)
        test_pred_tl = wsvm(X_train_new, y_train_new,X_test)
        sp,se,gmean = Gmean(y_test,test_pred_tl)
        metr(X_train,y_test,test_pred_tl,sp,se,gmean)
        metr_text(f,X_train,y_test,test_pred_tl,sp,se,gmean)
    f.write("\n===================================================================================\n")
    f.close()