# -*- coding: utf-8 -*-
import pandas as pd
from collections import Counter
import pickle,os,re
from keras.utils import to_categorical
import numpy as np
#from keras.callbacks import Callback
#from sklearn import metrics
from keras import backend as K
col_list = ['location_traffic_convenience', 'location_distance_from_business_district','location_easy_to_find',
            'service_wait_time', 'service_waiters_attitude', 'service_parking_convenience', 'service_serving_speed',
            'price_level', 'price_cost_effective', 'price_discount',
            'environment_decoration', 'environment_noise', 'environment_space', 'environment_cleaness',
            'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
            'others_overall_experience', 'others_willing_to_consume_again']


def F1_score(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def F1_macro(y_true, y_pred):
    # matthews_correlation
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + K.epsilon())

def to_true_label(y):
    y_pred = np.zeros((len(y)), dtype=np.int32)
    y_pred[y == 2] = -2
    y_pred[y == 3] = -1
    y_pred[y == 1] = 1
    return y_pred

def pickle_save(x,savefile=''):
    f = open(savefile+'.pkl', 'wb')
    pickle.dump(x, f, 0)
    f.close()

def pickle_load(input):
    f = open(input, 'rb')
    out = pickle.load(f)
    f.close()
    return out

def text_save(X,fl=''):
    f=open(fl,'w')
    f.writelines(X)
    f.close()
    print(fl,'save OK !')

def merge_vector(vec1='',vec2=''):
    v1 = pickle_load(input=vec1)
    v2 = pickle_load(input=vec2)
    print('char  num:', v1.__len__())
    print('word  num:', v2.__len__())
    char_word_dict={}
    for k,v in v1.items():
        char_word_dict[k]=v
    for k,v in v2.items():
        char_word_dict[k]=v
    print('char and word num:',char_word_dict.__len__())
    return char_word_dict

def merge_vector_line(vec1='data/w2v_word_skip_128_dict.pkl',
                      vec2='data/w2v_word_cbow_128_dict.pkl',savefile='data/w2v_word_skip_cbow_merge'):
    v1 = pickle_load(input=vec1)
    v2 = pickle_load(input=vec2)
    words=list(v1.keys())
    vec_dict={}
    for w in words:
        vec_dict[w]=v1[w]+v2[w]
    pickle_save(vec_dict, savefile)
    return vec_dict



