#! -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from Capsule2D_Keras import *
from keras.layers import *
from keras.models import Model,Sequential
from preprocess import *
from keras import utils
from keras.callbacks import ModelCheckpoint
from model_train import recall_threshold,precision_threshold
def CNN2D_train(x_train,y_train,x_test,y_test,model_name=''):
    y_test = utils.to_categorical(y_test)
    y_train = utils.to_categorical(y_train)
    n_classes=y_train.shape[1]
    input_vec = Input(shape=(None,None,1))
    cnn = Conv2D(128, (3, 32), activation='relu')(input_vec)
    #cnn = Conv2D(64, (3, 16), activation='relu')(cnn)
    cnn = MaxPooling2D((2,2))(cnn)
    cnn = Conv2D(128, (3, 32), activation='relu')(cnn)
    #cnn = Conv2D(128, (3, 16), activation='relu')(cnn)
    cnn = GlobalMaxPooling2D()(cnn)
    dense = Dense(128, activation='relu')(cnn)
    output = Dense(n_classes, activation='sigmoid')(dense)
    model = Model(inputs=input_vec, outputs=output)
    #model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2,optimizer='adam',metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',recall_threshold(0.5),precision_threshold(0.5)])
    model.summary()
    filepath = "model/CNN2D_" + model_name + "_{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, mode='auto')
    callbacks_list = [checkpoint]
    model.fit(x_train, y_train,batch_size=128, epochs=30,verbose=1,validation_data=(x_test, y_test),callbacks=callbacks_list)
    score, acc,recall,precision= model.evaluate(x_test, y_test, batch_size=128)
    print("\nTest score: %.4f, accuracy: %.4f, recall: %.4f,precision: %.4f" % (score, acc,recall,precision))
    print('&&end&&' * 10)

#搭建CNN+Capsule分类模型
def CNN2D_Capsule_train(x_train,y_train,x_test,y_test,model_name=''):
    y_test = utils.to_categorical(y_test)
    y_train = utils.to_categorical(y_train)
    n_classes=y_train.shape[1]
    input_vec = Input(shape=(None,None,1))
    cnn = Conv2D(128, (3, 32), activation='relu')(input_vec)
    #cnn = Conv2D(64, (3, 16), activation='relu')(cnn)
    cnn = MaxPooling2D((2,2))(cnn)
    #cnn = Conv2D(128, (3, 16), activation='relu')(cnn)
    cnn = Conv2D(128, (3, 32), activation='relu')(cnn)
    cnn = Reshape((-1, 128))(cnn)
    capsule = Capsule(n_classes, 16, 3, True)(cnn)
    output=Flatten()(capsule)
    output=Dense(n_classes, activation='sigmoid')(output)
    #output = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    model = Model(inputs=input_vec, outputs=output)
    #model.compile(loss=lambda y_true,y_pred: y_true*K.relu(0.9-y_pred)**2 + 0.25*(1-y_true)*K.relu(y_pred-0.1)**2,optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',recall_threshold(0.5),precision_threshold(0.5)])
    model.summary()
    filepath = "model/CNN2D_Capsul_" + model_name + "_{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, mode='auto')
    callbacks_list = [checkpoint]
    model.fit(x_train, y_train,batch_size=128,epochs=30,verbose=1,validation_data=(x_test, y_test),callbacks=callbacks_list)
    score, acc,recall,precision= model.evaluate(x_test, y_test, batch_size=128)
    print("\nTest score: %.4f, accuracy: %.4f, recall: %.4f,precision: %.4f" % (score, acc,recall,precision))
    print('&&end&&' * 10)
if __name__ == "__main__":
    max_text_word = 72
    max_num_words=70000
    data_file = 'data/comments_words_clean_sample.csv'
    word2vec_file = 'model/word2vec_skip_128.bin'
    print('train CNN2D model of score 1,5.....')
    x_train, y_train, x_test, y_test, embedding_matrix, embedding_layer = build_data(
        data_file=data_file, word2vec_file=word2vec_file, max_num_words=max_num_words,
        max_text_word=max_text_word, classes=[1, 5], vector=True)
    CNN2D_train(x_train, y_train, x_test, y_test, model_name='score_1_5')
    print('train CNN2D_Capsule model of score 1,5.....')
    CNN2D_Capsule_train(x_train, y_train, x_test, y_test, model_name='score_1_5')

    x_train, y_train, x_test, y_test, embedding_matrix, embedding_layer = build_data(
        data_file=data_file, word2vec_file=word2vec_file, max_num_words=max_num_words,
        max_text_word=max_text_word, classes=[1,3, 5], vector=True)
    print('train CNN2D model of score 1,3,5.....')
    CNN2D_train(x_train, y_train, x_test, y_test, model_name='score_1_3_5')
    print('train CNN2D_Capsule model of score 1,3,5.....')
    CNN2D_Capsule_train(x_train, y_train, x_test, y_test, model_name='score_1_3_5')

    x_train, y_train, x_test, y_test, embedding_matrix, embedding_layer = build_data(
        data_file=data_file, word2vec_file=word2vec_file, max_num_words=max_num_words,
        max_text_word=max_text_word, classes=[1, 2], vector=True)
    print('train CNN2D model of score 1,2.....')
    CNN2D_train(x_train, y_train, x_test, y_test, model_name='score_1_2')
    print('train CNN2D_Capsule model of score 1,2.....')
    CNN2D_Capsule_train(x_train, y_train, x_test, y_test, model_name='score_1_2')

    x_train, y_train, x_test, y_test, embedding_matrix, embedding_layer = build_data(
        data_file=data_file, word2vec_file=word2vec_file, max_num_words=max_num_words,
        max_text_word=max_text_word, classes=[2, 4], vector=True)
    print('train CNN2D model of score 2,4.....')
    CNN2D_train(x_train, y_train, x_test, y_test, model_name='score_2_4')
    print('train CNN2D_Capsule model of score 2,4.....')
    CNN2D_Capsule_train(x_train, y_train, x_test, y_test, model_name='score_2_4')

    x_train, y_train, x_test, y_test, embedding_matrix, embedding_layer = build_data(
        data_file=data_file, word2vec_file=word2vec_file, max_num_words=max_num_words,
        max_text_word=max_text_word, classes=[1,2,3,4, 5], vector=True)
    print('train CNN2D model of score 1,2,3,4,5.....')
    CNN2D_train(x_train, y_train, x_test, y_test, model_name='score_1_2_3_4_5')
    print('train CNN2D_Capsule model of score 1,2,3,4,5.....')
    CNN2D_Capsule_train(x_train, y_train, x_test, y_test, model_name='score_1_2_3_4_5')
