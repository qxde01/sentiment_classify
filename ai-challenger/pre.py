#! -*- coding: utf-8 -*-
import pandas  as pd
#import numpy as np
#import fasttext
#from tqdm import tqdm
from itertools import chain
from collections import Counter,OrderedDict
from joblib import Parallel, delayed
import jieba,re
from hanziconv import HanziConv
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from util import *
stopwords=open('dict/stopwords.txt').readlines()
stopwords=[w.strip() for w in  stopwords]
print('stopwords num:',len(stopwords))
def load_user_dict(dictfile='dict/dict_ext.txt'):
    dict_words=open(dictfile).readlines()
    dict_words=[w.strip() for w in dict_words if len(w.strip())>0]
    print('ext dict size:', len(dict_words))
    for w in dict_words:
        try:
            jieba.add_word(w,len(w)*10000,'nz')
        except Exception as e:
            print(e)

def seg(x):
    x=HanziConv.toSimplified(x)
    x = re.sub('\x05|\x06|\x07|\.\.|\.\.\.', ' ', x)
    #w = posseg.cut(x.upper())
    #w = [word for word, flag in w if word not in stopwords and flag in keep_property]
    w = jieba.cut(x.upper())
    w = [word.strip() for word in w if word not in stopwords and len(word.strip())>0]
    #w=[word.strip() for word in w if len(word)>0]
    return ' '.join(w)

def tochar(x):
    x=re.sub('\n|\t|\r| |"|。。|!!|…', ' ', x)
    x = re.sub('\n|\t|\r| |"|。。|!!', ' ', x)
    x=re.sub('\x05|\x06|\x07|\.\.|\.\.\.',' ',x)
    x = HanziConv.toSimplified(x)
    x=list(x.strip())
    x=[a for a in x if len(a.strip())>0]
    return ' '.join(x)

def build_corpus():
    X3 = pd.read_csv('rawdata/sentiment_analysis_testa.csv')
    X2 = pd.read_csv('rawdata/sentiment_analysis_validationset.csv')
    X1 = pd.read_csv('rawdata/sentiment_analysis_trainingset.csv')
    X = pd.concat([X1, X2, X3], axis=0)
    del (X1, X2, X3)
    words = Parallel(n_jobs=18, verbose=1, pre_dispatch='2*n_jobs')(delayed(seg)(x) for x in X.content)
    char = Parallel(n_jobs=10, verbose=1, pre_dispatch='2*n_jobs')(delayed(tochar)(x) for x in X.content)
    X['words']=words
    X['char']=char
    X.to_csv('data/corpus.csv',index=False,encoding='UTF-8')
    return X

def words_freq(words):
    w = [x.split(' ') for x in words]
    w = list(chain.from_iterable(w))
    f = Counter(w)
    f = [(k, v) for k, v in f.items()]
    f = sorted(f, key=lambda x: x[1], reverse=True)
    print('words num:',len(f))
    f1=[k for k ,v in f if v>1]
    f2 = [k for k, v in f if v > 2]
    print('words num and min freq = 2:', len(f1))
    print('words num and min freq = 3:', len(f2))
    return f

def get_sentence(x):
    x = re.sub('"', ' ', x)
    x = re.split('[\n\r\t。！]|…', x)
    x = [HanziConv.toSimplified(a.strip()) for a in x if len(a.strip()) > 0]
    return x

def build_sentences():
    X = pd.read_csv('data/corpus.csv')
    sentences = Parallel(n_jobs=10, verbose=1, pre_dispatch='2*n_jobs')(delayed(get_sentence)(x) for x in X.content)
    sentences = list(chain.from_iterable(sentences))
    sentences = [x.strip() + '\n' for x in sentences if len(x.strip()) > 1]
    print('sentences num:', len(sentences))
    text_save(sentences, fl='data/sentences.txt')
    s_words = Parallel(n_jobs=18, verbose=1, pre_dispatch='2*n_jobs')(delayed(seg)(x) for x in sentences)
    s_words = [x + '\n' for x in s_words]
    text_save(s_words, fl='data/sentences_words.txt')
    s_char = Parallel(n_jobs=10, verbose=1, pre_dispatch='2*n_jobs')(delayed(tochar)(x) for x in sentences)
    s_char = [x + '\n' for x in s_char]
    text_save(s_char, fl='data/sentences_char.txt')

def build_data(max_num_word=210000,max_len_word=600,max_num_char=7100,max_len_char=1500):
    X = pd.read_csv('data/corpus.csv')
    max_len_word1 = max([len(x.split(' ')) for x in X.words])
    max_len_word=min(max_len_word,max_len_word1)
    print('text max words num:%d,只取:%d'%(max_len_word1,max_len_word) )
    max_len_char1 = max([len(x.split(' ')) for x in X.char])
    max_len_char = min(max_len_char, max_len_char1)
    print('text max char num:%d,只取:%d'%(max_len_char1,max_len_char))
    words_tokenizer = Tokenizer(num_words=max_num_word)
    words_tokenizer.fit_on_texts(X.words)
    words_sequences = words_tokenizer.texts_to_sequences(X.words)
    words_seq = pad_sequences(words_sequences, maxlen=max_len_word)
    pickle_save(words_seq, savefile='data/words_seq_' + str(max_num_word)+'_'+str(max_len_word))
    print('Found %s unique tokens.' % len(words_tokenizer.word_index))
    char_tokenizer = Tokenizer(num_words=max_num_char)
    char_tokenizer.fit_on_texts(X.char)
    char_sequences = char_tokenizer.texts_to_sequences(X.char)
    char_seq = pad_sequences(char_sequences, maxlen=max_len_char)
    pickle_save(char_seq, savefile='data/char_seq_' + str(max_num_char)+'_'+str(max_len_char))
    print('Found %s unique char.' % len(char_tokenizer.word_index))

def get_data(max_num_word=210000,max_len_word=600,level='words'):
    X = pd.read_csv('data/corpus.csv')
    #max_len_word = max([len(x.split(' ')) for x in X.words])
    #print('text max words num:', max_len_word)
    if level=='words':
        words_seq1 = pickle_load('data/words_seq_' + str(max_num_word) +'_'+str(max_len_word)+ '.pkl')
    else:
        words_seq1 = pickle_load('data/char_seq_' + str(max_num_word) + '_' + str(max_len_word) + '.pkl')
    # train_data = X.loc[:105000]
    x_val1 = words_seq1[105000:120000]
    #words_seq2 = pickle_load('data/word_seq_256.pkl')
    #x_val2 = words_seq2[105000:120000]
    x_train1 = words_seq1[:105000]
    #x_train2 = words_seq2[:105000]
    y_train = OrderedDict()
    y_val = OrderedDict()
    for col in col_list:
        y_train[col] = to_categorical(X[col][:105000], num_classes=4)
        y_val[col] = to_categorical(X[col][105000:120000], num_classes=4)
    x_train={'input1': x_train1}
    x_val={'input1': x_val1}
    x_test={'input1': words_seq1[120000:]}
    return x_train,y_train,x_val,y_val,x_test

def get_data_sample(max_num_word=210000,max_len_word=600,level='words',col='',maxN=30000):
    X = pd.read_csv('data/corpus.csv')
    if level=='words':
        words_seq1 = pickle_load('data/words_seq_' + str(max_num_word) +'_'+str(max_len_word)+ '.pkl')
    else:
        words_seq1 = pickle_load('data/char_seq_' + str(max_num_word) + '_' + str(max_len_word) + '.pkl')
    train_X = X.loc[:105000]
    #n = train_X.shape[0]
    cl = Counter(train_X[col])
    cl = [(k, v) for k, v in cl.items()]
    maxN = min(maxN, max([v for k, v in cl]))
    print(col, cl)
    sam = []
    for k, v in cl:
        rate = float(v / maxN)
        x0 = train_X[train_X[col] == k]
        print(k, v, rate)
        if rate < 1.0:
            x0 = x0.sample(frac=1. / rate, replace=True).reset_index(drop=True)
        else:
            x0 = x0.sample(frac=1.0 / rate).reset_index(drop=True)
        sam.append(x0)
    Z = pd.concat(sam)
    Z = Z.sample(frac=1).reset_index(drop=True)
    Z = Z.sample(frac=1).reset_index(drop=True)
    x_val1 = words_seq1[105000:120000]
    x_train1 = words_seq1[:105000]
    x_train1 = x_train1[Z.id]
    y_train = OrderedDict()
    y_val = OrderedDict()
    for col in col_list:
        print(Counter(Z[col]),'||',col)
        y_train[col] = to_categorical(Z[col], num_classes=4)
        y_val[col] = to_categorical(X[col][105000:120000], num_classes=4)
    x_train={'input1': x_train1}
    x_val={'input1': x_val1}
    x_test={'input1': words_seq1[120000:]}
    print('X data shape:',x_train1.shape)
    return x_train,y_train,x_val,y_val,x_test