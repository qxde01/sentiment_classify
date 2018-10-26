#! -*- coding: utf-8 -*-
import pandas  as pd
import numpy as np
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras.layers import Embedding
from keras.utils import to_categorical
from keras.layers import *
from keras import optimizers
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from util import *
from collections import OrderedDict
from Capsule_Keras import *
from attention_keras import *
from pre import get_data

def build_CNN1D(max_num_word=210000,max_len_word=600):
    input1 = Input(shape=(max_len_word,), dtype='int32', name='input1')
    x0 = Embedding(output_dim=300,input_dim=max_num_word, input_length=max_len_word,name='embedding_0')(input1)
    x0 = Dropout(0.45)(x0)
    x0 = Conv1D(256,5,padding='valid', activation='elu',strides=1)(x0)
    output=[]
    for col in col_list:
        z1 = Conv1D(128, 5, padding='valid', activation='elu', strides=1)(x0)
        z2 = Conv1D(128, 3, padding='valid', activation='elu', strides=1)(x0)
        x1 = GlobalMaxPooling1D()(z1)
        x2 = GlobalAveragePooling1D()(z1)
        x3 = GlobalMaxPooling1D()(z2)
        x4 = GlobalAveragePooling1D()(z2)
        x = concatenate([x1, x2, x3, x4])
        x = Dense(256, activation='elu')(x)
        output.append(Dense(4, activation='softmax', name=col)(x))
    model = Model(inputs=[input1], outputs=output)
    model.summary()
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc', F1_macro])
    filepath = "model/CNN1D_{epoch:03d}-{val_loss:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, save_best_only=False, verbose=1, mode='auto')
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='auto',epsilon=0.0001, cooldown=0, min_lr=0.000000001)
    callbacks_list = [checkpoint,reducelr]
    return model, callbacks_list

def build_GRU_Capsule(max_num_word=210000,max_len_word=600):
    input1 = Input(shape=(max_len_word,), dtype='int32', name='input1')
    x1 = Embedding(output_dim=64, input_dim=max_num_word, input_length=max_len_word, name='embedding_0')(input1)
    x1=SpatialDropout1D(0.35)(x1)
    bigru = Bidirectional(GRU(64, dropout=0.45, recurrent_dropout=0.35, return_sequences=True), input_shape=(max_len_word, 64))(x1)
    output=[]
    for col in col_list:
        z=Capsule(num_capsule=10, dim_capsule=16, routings=3, share_weights=True)(bigru)
        z=Flatten()(z)
        z=Dropout(0.1)(z)
        output.append(Dense(4, activation='softmax', name=col)(z))
    #x=Dropout(0.35)(x)
    model = Model(inputs=[input1], outputs=output)
    model.summary()
    adam = optimizers.Adam(lr=0.0015, beta_1=0.99, beta_2=0.99999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc', F1_macro])
    filepath = "model/gru_caps_{epoch:03d}-{val_loss:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, save_best_only=False, verbose=1, mode='auto')
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='min', epsilon=0.0001,cooldown=0, min_lr=0.00000001)
    es=EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    callbacks_list = [checkpoint, reducelr,es]
    return model, callbacks_list

def build_CNN_LSTM(max_num_word=210000,max_len_word=600):
    input1 = Input(shape=(max_len_word,), dtype='int32', name='input1')
    x1 = Embedding(output_dim=256, input_dim=max_num_word, input_length=max_len_word, name='embedding_0')(input1)
    x1=Conv1D(256, 3, padding='valid', activation='elu', strides=1)(x1)
    output = []
    for col in col_list:
        #z=LSTM(128,dropout=0.25, recurrent_dropout=0.25)(x1)
        z = CuDNNLSTM(128)(x1)
        output.append(Dense(4, activation='softmax', name=col)(z))
    model = Model(inputs=[input1], outputs=output)
    adam = optimizers.Adam(lr=0.0012, beta_1=0.99, beta_2=0.99999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy',F1_macro])
    model.summary()
    filepath="model/CNN_LSTM_{epoch:03d}-{val_loss:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath,save_best_only=False, verbose=1, mode='auto')
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.45, patience=3, verbose=1, mode='min', epsilon=0.0001,cooldown=0, min_lr=0.00000001)
    callbacks_list = [checkpoint,reducelr]
    return model, callbacks_list

def build_biLSTM(max_num_word=210000,max_len_word=600):
    input1 = Input(shape=(max_len_word,), dtype='int32', name='input1')
    x1 = Embedding(output_dim=128, input_dim=max_num_word, input_length=max_len_word, name='embedding_0')(input1)
    #x2 = Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.45), input_shape=(max_len_word, 256))(x2)
    x1=SpatialDropout1D(0.28)(x1)
    x1=Bidirectional(CuDNNLSTM(128, return_sequences=True), input_shape=(max_len_word, 128))(x1)
    #model.add(Bidirectional(LSTM(128,dropout=0.25, recurrent_dropout=0.25)))
    output = []
    for col in col_list:
        #z=LSTM(128,dropout=0.25, recurrent_dropout=0.25)(x1)
        #z = CuDNNLSTM(128)(x1)
        z=Bidirectional(CuDNNLSTM(128))(x1)
        output.append(Dense(4, activation='softmax', name=col)(z))
    model = Model(inputs=[input1], outputs=output)
    adam = optimizers.Adam(lr=0.0012, beta_1=0.99, beta_2=0.99999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy',F1_macro])
    model.summary()
    filepath="model/biLSTM_{epoch:03d}-{val_loss:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath,save_best_only=False, verbose=1, mode='auto')
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.45, patience=3, verbose=1, mode='min', epsilon=0.0001,cooldown=0, min_lr=0.00000001)
    callbacks_list = [checkpoint,reducelr]
    return model, callbacks_list

def build_biGRU(max_num_word=210000,max_len_word=600):
    input1 = Input(shape=(max_len_word,), dtype='int32', name='input1')
    x1 = Embedding(output_dim=128, input_dim=max_num_word, input_length=max_len_word, name='embedding_0')(input1)
    #x2 = Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.45), input_shape=(max_len_word, 256))(x2)
    x1=SpatialDropout1D(0.3)(x1)
    x1=Bidirectional(CuDNNGRU(128, return_sequences=True), input_shape=(max_len_word, 128))(x1)
    #model.add(Bidirectional(LSTM(128,dropout=0.25, recurrent_dropout=0.25)))
    output = []
    for col in col_list:
        #z=LSTM(128,dropout=0.25, recurrent_dropout=0.25)(x1)
        #z = CuDNNLSTM(128)(x1)
        z=Bidirectional(CuDNNGRU(100))(x1)
        output.append(Dense(4, activation='softmax', name=col)(z))
    model = Model(inputs=[input1], outputs=output)
    adam = optimizers.Adam(lr=0.0012, beta_1=0.99, beta_2=0.99999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy',F1_macro])
    model.summary()
    filepath="model/biGRU_{epoch:03d}-{val_loss:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath,save_best_only=False, verbose=1, mode='auto')
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.45, patience=2, verbose=1, mode='min', epsilon=0.0001,cooldown=0, min_lr=0.00000001)
    callbacks_list = [checkpoint,reducelr]
    return model, callbacks_list


def build_attention(max_num_word=180000,max_len_word=901):
    input1 = Input(shape=(max_len_word,), dtype='int32', name='input1')
    x1 = Embedding(output_dim=128, input_dim=max_num_word, input_length=max_len_word, name='embedding_0')(input1)
    #x1 = Position_Embedding()(x1) # 增加Position_Embedding能轻微提高准确率
    #x = Dropout(0.5)(x)
    output = []
    for col in col_list:
        x = Attention(8, 16)([x1, x1,x1])
        x1 = GlobalAveragePooling1D()(x)
        #x2 = GlobalMaxPooling1D()(x)
        #z = concatenate([Flatten()(x1), Flatten()(x2)])
        output.append(Dense(4, activation='softmax', name=col)(x1))
    model = Model(inputs=[input1], outputs=output)
    model.summary()
    adam = optimizers.Adam(lr=0.0015, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc', F1_macro])
    filepath = "model/atten_{epoch:03d}-{val_loss:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, save_best_only=False, verbose=1, mode='auto')
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    callbacks_list = [checkpoint, reducelr]
    return model, callbacks_list
#model, callbacks_list=build_attention(max_num_word=7200,max_len_word=1500)

if __name__ == "__main__":
    x_train,y_train,x_val,y_val,x_test=get_data(max_num_word=7400,max_len_word=1200,level='char')
    model, callbacks_list = build_GRU_Capsule(max_num_word=7400, max_len_word=1200)
    #model, callbacks_list=build_biGRU(max_num_word=7200,max_len_word=1500)
    #from keras.utils import plot_model
    #plot_model(model, to_file='model.png')
    model.fit(x_train, y_train, epochs=15, batch_size=32, validation_data=(x_val, y_val), callbacks=callbacks_list)
    #sgd=optimizers.SGD(lr=0.0005, momentum=0.9, decay=0.000001, nesterov=True)
    adam = optimizers.Adam(lr=0.0005, beta_1=0.99, beta_2=0.99999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy',F1_macro])
    model.fit(x_train , y_train, epochs=10, batch_size=64,validation_data=(x_val ,  y_val), callbacks=callbacks_list)