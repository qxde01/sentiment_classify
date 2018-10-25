# -*- coding: utf-8 -*-
from util import *
from tqdm import  tqdm
import pandas as pd
from keras.layers import Embedding
import numpy as np
from joblib import Parallel, delayed

def get_text_id(idx,x):
    x = x.split(' ')
    m = len(x)
    #print(m)
    line  =np.zeros(max_text_word,dtype=np.int32)
    for j in range(0, m):
        if j < max_text_word:
            if w2v_words_list.__contains__(x[j]):
                line[j]=words_dict[x[j]]
    return idx,line

def build_corpus_vector(data_file='data/corpus_validationset.csv',word2vec_file='data/w2v_word_skip_128_dict.pkl',
                        embeddings_dim=128,level='word',vector=False):
    w2v_vector = pickle_load(word2vec_file)
    data=pd.read_csv(data_file)
    n = data.shape[0]
    global max_text_word
    #max_text_word=1140
    if level == 'char':
        data = data.loc[:, ['id', 'char']]
        data.columns = ['id', 'words']
        #max_text_word = max([len(x.split(' ')) for x in data.char])
    else :
        data = data.loc[:, ['id', 'words']]
        #data.columns = ['id', 'word']
    max_text_word = max([len(x.split(' ')) for x in data.words])
    global w2v_words_list,words_dict
    w2v_words_list=list(w2v_vector.keys())
    words_dict={}
    w2v_words_num=len(w2v_words_list)
    print('words number:',w2v_words_num)
    print('max_text_word:',max_text_word)
    embedding_matrix = np.zeros((w2v_words_num+1, embeddings_dim))
    for i in tqdm(range(0,w2v_words_num)):
        words_dict[w2v_words_list[i]]=i+1
        embedding_matrix[i+1] = w2v_vector[w2v_words_list[i]]
    print('embedding matrix shape:', embedding_matrix.shape )
    embedding_layer = Embedding(w2v_words_num+1, embeddings_dim,weights=[embedding_matrix], input_length=max_text_word, trainable=False)
    data_seq = Parallel(n_jobs=20, verbose=1, pre_dispatch='2*n_jobs')(delayed(get_text_id)(data.id[x], data.words[x]) for x in range(n))
    data_seq = sorted(data_seq, key=lambda x: x[0], reverse=False)
    data_seq =[x[1] for x in data_seq]
    data_seq =np.array(data_seq)
    print('Shape of data tensor:', data_seq.shape)
    ## for CNN
    if vector==True:
        X_train=np.zeros((n, max_text_word, embeddings_dim), dtype=np.float32)
        print('gen train data vector:')
        for i in tqdm(range(0,n)):
            z=data_seq[i,:]
            for j in range(0,max_text_word):
                    X_train[i, j, :] = embedding_matrix[z[j]]
        X_train = X_train.reshape(n, max_text_word, embeddings_dim, 1)
        return X_train,embedding_matrix,embedding_layer
    else:
        return data_seq,embedding_matrix,embedding_layer

if __name__ == "__main__":
    data_seq, embedding_matrix, embedding_layer=build_corpus_vector(data_file='data/corpus.csv', word2vec_file='data/w2v_word_skip_256_dict.pkl',
                        embeddings_dim=256, level='word', vector=False)
    pickle_save(data_seq,'data/word_seq_256')
    pickle_save(embedding_layer, 'data/word_embedding_layer_256')

    data_seq, embedding_matrix, embedding_layer = build_corpus_vector(data_file='data/corpus.csv',
                                                                      word2vec_file='data/w2v_char_skip_256_dict.pkl',
                                                                      embeddings_dim=256, level='char', vector=False)
    pickle_save(data_seq, 'data/char_seq_256')
    pickle_save(embedding_layer, 'data/char_embedding_layer_256')