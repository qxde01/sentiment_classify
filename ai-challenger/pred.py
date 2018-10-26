# -*- coding: utf-8 -*-
from keras.models import load_model
import numpy as np
from sklearn import metrics
#from pycm import *
from util import *
import pandas as pd
from pre import get_data

def model_val(X,x_val,model_path):
    model = load_model(model_path, custom_objects={'F1_macro': F1_macro})
    y_pred = model.predict(x_val, batch_size=80, verbose=1)
    F1 = []
    n=len(col_list)
    for i in range(0, n):
        y0 = np.argmax(y_pred[i], axis=1)
        y0 = to_true_label(y0)
        y_true = X[col_list[i]]
        print(metrics.confusion_matrix(y_true, y0))
        f11 = metrics.f1_score(y_true, y0, average='macro')
        print(col_list[i], f11)
        F1.append((col_list[i],f11))
    F11=[x[1] for x in F1]
    print('sklearn 4 classes avg macro F1:', sum(F11) / len(F11))
    del(model)
    return y_pred, F1

def capse_val(X,x_val,model_path):
    model = load_model(model_path, custom_objects={'F1_macro': F1_macro,'Capsule':Capsule})
    y_pred = model.predict(x_val, batch_size=128, verbose=1)
    F1 = []
    n=len(col_list)
    for i in range(0, n):
        y0 = np.argmax(y_pred[i], axis=1)
        y0 = to_true_label(y0)
        y_true = X[col_list[i]]
        print(metrics.confusion_matrix(y_true, y0))
        f11 = metrics.f1_score(y_true, y0, average='macro')
        print(col_list[i], f11)
        F1.append((col_list[i], f11))
    F11 = [x[1] for x in F1]
    print('sklearn 4 classes avg macro F1:', sum(F11) / len(F11))
    del(model)
    return y_pred, F1

def atten_val(X,x_val,model_path,col_list=col_list):
    model = load_model(model_path, custom_objects={'F1_macro': F1_macro,'Attention':Attention,'Position_Embedding':Position_Embedding})
    y_pred = model.predict(x_val, batch_size=1024, verbose=1)
    F1 = []
    n=len(col_list)
    for i in range(0, n):
        y0 = np.argmax(y_pred[i], axis=1)
        y0 = to_true_label(y0)
        y_true = X[col_list[i]]
        print(metrics.confusion_matrix(y_true, y0))
        f11 = metrics.f1_score(y_true, y0, average='macro')
        print(col_list[i], f11)
        F1.append(f11)
    print('sklearn 4 classes avg macro F1:', sum(F1) / len(F1))
    return y_pred, F1

def test_pred(x_test,model_path='biGRU'):
    X=pd.read_csv('rawdata/sentiment_analysis_testa.csv')
    model = load_model(model_path, custom_objects={'F1_macro': F1_macro})
    y_pred = model.predict(x_test, batch_size=64, verbose=1)
    for i in range(0,20):
        y0 = np.argmax(y_pred[i], axis=1)
        y0 = to_true_label(y0)
        X[col_list[i]]=y0
    X.to_csv('resu_testa_1024.csv',index=False,encoding='UTF-8')
    return X

if __name__ == "__main__":
    X = pd.read_csv('rawdata/sentiment_analysis_validationset.csv')
    x_train, y_train, x_val, y_val, x_test = get_data(max_num_word=7200, max_len_word=1500, level='char')
    resu = model_val(X=X, x_val=x_val, model_path='model/CNN1D_GRU_010-14.1442.h5')