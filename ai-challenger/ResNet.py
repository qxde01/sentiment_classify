#! -*- coding: utf-8 -*-
from keras.layers import *
from keras import optimizers
from keras.models import Model
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from pre import get_data
from util import *

def Conv1d_BN(x, nb_filter, kernel_size, strides=1, padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv1D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=-1, name=bn_name)(x)
    return x

def identity_Block(x, nb_filter, kernel_size, strides=1, with_conv_shortcut=False):
    x1 = Conv1d_BN(x,nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x1 = Conv1d_BN(x1,nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv1d_BN(x1,nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x1, shortcut])
        return x
    else:
        x = add([x1, x])
        return x

def bottleneck_Block(x,nb_filters,strides=1,with_conv_shortcut=False):
    k1,k2,k3=nb_filters
    x1 = Conv1d_BN(x,nb_filter=k1, kernel_size=1, strides=strides, padding='same')
    x1 = Conv1d_BN(x1,nb_filter=k2, kernel_size=3, padding='same')
    x1 = Conv1d_BN(x1,nb_filter=k3, kernel_size=1, padding='same')
    if with_conv_shortcut:
        shortcut = Conv1d_BN(x1,nb_filter=k3, strides=strides, kernel_size=1)
        x = add([x1, shortcut])
        return x
    else:
        x = add([x1, x])
        return x

def build_resnet(max_num_word=210000,max_len_word=600):
    input1 = Input(shape=(max_len_word,), dtype='int32', name='input1')
    x1 = Embedding(output_dim=64, input_dim=max_num_word, input_length=max_len_word, name='embedding_0')(input1)
    #conv1
    x = Conv1d_BN(x1,nb_filter=64, kernel_size=2, strides=1, padding='valid')
    #x = MaxPooling1D(pool_size=2, padding='same')(x)
    output = []
    for col in col_list:
        z = identity_Block(x,nb_filter=64, kernel_size=2)
        z = identity_Block(z,nb_filter=64, kernel_size=2, with_conv_shortcut=True)
        z = identity_Block(z,nb_filter=64, kernel_size=2)
        z = identity_Block(z,nb_filter=64, kernel_size=2, with_conv_shortcut=True)
        z = identity_Block(z,nb_filter=64, kernel_size=2)
        z1 = GlobalMaxPooling1D()(z)
        z2 = GlobalAveragePooling1D()(z)
        zo = concatenate([z1, z2])
        zo=Dropout(0.25)(zo)
        zo = Dense(64, activation='relu')(zo)
        output.append(Dense(4, activation='softmax', name=col)(zo))
    model = Model(inputs=[input1], outputs=output)
    model.summary()
    adam = optimizers.Adam(lr=0.0015, beta_1=0.99, beta_2=0.99999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc', F1_macro])
    filepath = "model/resnet_{epoch:03d}-{val_loss:.4f}.h5"
    checkpoint = ModelCheckpoint(filepath, save_best_only=False, verbose=1, mode='auto',save_weights_only=True)
    reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='min', epsilon=0.0001,
                                 cooldown=0, min_lr=0.00000001)
    es = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    callbacks_list = [checkpoint, reducelr, es]
    return model, callbacks_list
if __name__ == "__main__":
    x_train,y_train,x_val,y_val,x_test=get_data(max_num_word=7400,max_len_word=1200,level='char')
    model, callbacks_list=build_resnet(max_num_word=7400,max_len_word=1200)
    model.fit(x_train, y_train, epochs=15, batch_size=32, validation_data=(x_val, y_val), callbacks=callbacks_list)
