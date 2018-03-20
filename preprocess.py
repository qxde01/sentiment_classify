#! -*- coding: utf-8 -*-
import pandas  as pd
import numpy as np
import fasttext
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding

def build_data(data_file='data/comments_words_sample.csv',
               word2vec_file='model/word2vec_skip_128.bin',
               max_num_words=20000,max_text_word=80,classes=[1,2],vector=False):
    w2v_model=fasttext.load_model(word2vec_file)
    data=pd.read_csv(data_file)
    if classes is not None:
        data=data[data.score.isin(classes)]
        classes2=[i for i in range(0,len(classes))]
        labels=np.zeros(data.shape[0])
        for k in range(0,len(classes)):
            labels[data.score==classes[k]]=classes2[k]
    else:
        labels=np.asarray(data.score - min(data.score))
    embeddings_dim=w2v_model.dim
    w2v_words_list=list(w2v_model.words)
    w2v_words_num=len(w2v_model.words)
    texts=data.words.tolist()
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    print('word2vec  has %s tokens.' % w2v_words_num)
    data_seq = pad_sequences(sequences, maxlen=max_text_word)
    #labels = to_categorical(np.asarray(data.score - min(data.score)))
    print('Shape of data tensor:', data_seq.shape)
    print('Shape of label tensor:', labels.shape)
    indices = np.arange(data_seq.shape[0])
    np.random.shuffle(indices)
    data_seq = data_seq[indices]
    labels = labels[indices]
    num_validation_samples = int(0.2 * data.shape[0])
    x_train = data_seq[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_test = data_seq[-num_validation_samples:]
    y_test = labels[-num_validation_samples:]
    num_words = min(max_num_words, w2v_words_num+ 1,len(word_index)+1)
    embedding_matrix = np.zeros((num_words, embeddings_dim))
    for word, i in word_index.items():
        if i >= max_num_words:
            continue
        #embedding_vector = w2v_model(word)
        if w2v_words_list.__contains__(word) :
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = w2v_model[word]
        if i %1001==0:
            print('process words:',i)
    print('embedding matrix shape:',(num_words, embeddings_dim))
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                embeddings_dim,
                                weights=[embedding_matrix],
                                input_length=max_text_word,
                                trainable=False)
    if vector==True:
        n1=x_train.shape[0]
        n2=x_test.shape[0]
        X_train=np.zeros((n1, max_text_word, embeddings_dim), dtype=np.float32)
        X_test = np.zeros((n2, max_text_word, embeddings_dim), dtype=np.float32)
        for i in range(0,n1):
            z=x_train[i,:]
            for j in range(0,max_text_word):
                if z[j]>0:
                    X_train[i, j, :] = embedding_matrix[z[j]]
            if i% 1001==0:
                print('gen train data vector:', i + 1)
        for i in range(0,n2):
            z=x_test[i,:]
            for j in range(0,max_text_word):
                if z[j]>0:
                    X_test[i, j, :] = embedding_matrix[z[j]]
            if i% 1001==0:
                print('gen test data vector:', i + 1)
        X_train = X_train.reshape(n1, max_text_word, embeddings_dim, 1)
        X_test = X_test.reshape(n2, max_text_word, embeddings_dim, 1)
        return X_train,y_train,X_test,y_test,embedding_matrix,embedding_layer
    else:
        return x_train,y_train,x_test,y_test,embedding_matrix,embedding_layer

