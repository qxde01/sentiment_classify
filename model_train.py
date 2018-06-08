#! -*- coding: utf-8 -*-
from keras.utils import to_categorical
from keras.layers import *
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers.wrappers import Bidirectional,TimeDistributed
from sklearn import metrics
from keras import backend as K
from Capsule_Keras import *
from attention_keras import *
# train a 1D convnet with global maxpooling

def recall_threshold(threshold = 0.5):
    def recall(y_true, y_pred):
        """Recall metric.
        Computes the recall over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio
    return recall

def precision_threshold(threshold=0.5):
    def precision(y_true, y_pred):
        """Precision metric.
        Computes the precision over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # count the predicted positives
        predicted_positives = K.sum(y_pred)
        # Get the precision ratio
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio
    return precision


def CNN1D_train(x_train,y_train,x_test,y_test,max_text_word=32,embedding_layer='',model_name=''):
    y_test = to_categorical(y_test)
    y_train = to_categorical(y_train)
    num_classes = y_train.shape[1]
    sequence_input = Input(shape=(max_text_word,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    #x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    #x = MaxPooling1D(5)(x)
    x=Dropout(0.5)(x)
    x = Conv1D(64, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(num_classes,activation='sigmoid')(x)
    filepath="model/CNN1D_"+model_name+"_{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, save_best_only=True,verbose=1, mode='auto')
    callbacks_list = [checkpoint]
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['acc',recall_threshold(0.5),precision_threshold(0.5)])
    model.summary()
    model.fit(x_train, y_train,batch_size=128,epochs=30,validation_data=(x_test, y_test),callbacks=callbacks_list)
    score, acc,recall,precision= model.evaluate(x_test, y_test, batch_size=128)
    print("\nTest score: %.4f, accuracy: %.4f, recall: %.4f,precision: %.4f" % (score, acc,recall,precision))
    print('&&end&&' * 10)

def LSTM_train(x_train,y_train,x_test,y_test,embedding_layer='',model_name=''):
    y_test = to_categorical(y_test)
    y_train = to_categorical(y_train)
    num_classes = y_train.shape[1]
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(128, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(num_classes, activation='sigmoid'))
    filepath="model/LSTM_"+model_name+"_{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, save_best_only=True,verbose=1, mode='auto')
    callbacks_list = [checkpoint]
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc',recall_threshold(0.5),precision_threshold(0.5)])
    model.fit(x_train, y_train,batch_size=128,epochs=30,validation_data=(x_test, y_test),callbacks=callbacks_list)
    score, acc,recall,precision= model.evaluate(x_test, y_test, batch_size=128)
    print("\nTest score: %.4f, accuracy: %.4f, recall: %.4f,precision: %.4f" % (score, acc,recall,precision))
    print('&&end&&' * 10)

#########################################################################
def CNN_LSTM_train(x_train,y_train,x_test,y_test,embedding_layer='',model_name=''):
    y_test = to_categorical(y_test)
    y_train = to_categorical(y_train)
    num_classes = y_train.shape[1]
    model = Sequential()
    #model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(embedding_layer)
    #model.add(Dropout(0.25))
    model.add(Conv1D(256,5,padding='valid',activation='elu',strides=1))
    #model.add(MaxPooling1D(pool_size=5))
    model.add(LSTM(128,dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(num_classes,activation='sigmoid'))
    #model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy',recall_threshold(0.5),precision_threshold(0.5)])
    model.summary()
    filepath="model/LSTM_CNN_"+model_name+"_{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath,save_best_only=True, verbose=1, mode='auto')
    callbacks_list = [checkpoint]
    print('Train...')
    model.fit(x_train, y_train,batch_size=128, epochs=30, validation_data=(x_test, y_test),callbacks=callbacks_list)
    score, acc,recall,precision= model.evaluate(x_test, y_test, batch_size=128)
    print("\nTest score: %.4f, accuracy: %.4f, recall: %.4f,precision: %.4f" % (score, acc,recall,precision))
    print('&&end&&' * 10)

def test_predict(model,x_test,y_test):
    pred=[]
    for x in x_test:
        pred.append(model.predict_classes(x.reshape((1,x.shape[0])))[0])
    pred = np.array(pred, dtype=np.int32)
    Recall = metrics.recall_score(y_test, pred, average='micro')
    Precision = metrics.accuracy_score(y_test, pred)
    F1_score = metrics.f1_score(y_test, pred,average='micro')
    Fbeta_score=metrics.fbeta_score(y_test, pred, average='micro', beta=0.5)
    print(">>>>> Precision:%.4f, Recall:%.4f, F1_score:%.4f,Fbeta_score:% .4f"%(Precision ,Recall,F1_score,Fbeta_score))
    Confusion = metrics.confusion_matrix(y_test,pred,)
    print('Confusion:\n',Confusion)

def biLSTM_train(x_train,y_train,x_test,y_test,input_shape=(72, 128),embedding_layer='',model_name=''):
    y_test0 = to_categorical(y_test)
    y_train = to_categorical(y_train)
    num_classes=y_train.shape[1]
    model = Sequential()
    model.add(embedding_layer)
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(Bidirectional(LSTM(128,dropout=0.25, recurrent_dropout=0.25)))
    model.add(Dense(num_classes,activation='sigmoid'))
    #model.add(Activation('softmax'))
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',recall_threshold(0.5),precision_threshold(0.5)])
    model.summary()
    filepath="model/biLSTM_"+model_name+"_{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, save_best_only=True,verbose=1, mode='auto')
    callbacks_list = [checkpoint]
    #print('Train...')
    model.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_test, y_test0), callbacks=callbacks_list)
    score, acc,recall,precision= model.evaluate(x_test, y_test, batch_size=128)
    print("\nTest score: %.4f, accuracy: %.4f, recall: %.4f,precision: %.4f" % (score, acc,recall,precision))
    print('&&end&&'*10)

def GRU_Capsule_train(x_train,y_train,x_test,y_test,gru_len=128,embedding_layer='',model_name=''):
    y_test = to_categorical(y_test)
    y_train = to_categorical(y_train)
    num_classes=y_train.shape[1]
    model = Sequential()
    model.add(embedding_layer)
    model.add(SpatialDropout1D(0.28))
    model.add(Bidirectional( GRU(gru_len, activation='elu', dropout=0.25,recurrent_dropout=0.25, return_sequences=True)))
    model.add(Capsule(num_capsule=num_classes, dim_capsule=16, routings=3, share_weights=True))
    model.add(Flatten())
    #model.add(Dropout(0,25))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',recall_threshold(0.5),precision_threshold(0.5)])
    model.summary()
    filepath = "model/gru_capsule_" + model_name + "_{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath,save_best_only=True, verbose=1, mode='auto')
    callbacks_list = [checkpoint]
    model.fit(x_train, y_train, batch_size=128, epochs=3, verbose=1, validation_data=(x_test, y_test),callbacks=callbacks_list)
    score, acc,recall,precision= model.evaluate(x_test, y_test, batch_size=128)
    print("\nTest score: %.4f, accuracy: %.4f, recall: %.4f,precision: %.4f" % (score, acc,recall,precision))
    print('&&end&&' * 10)

def attention_train(x_train,y_train,x_test,y_test,embedding_layer='',model_name=''):
    y_test = to_categorical(y_test)
    y_train = to_categorical(y_train)
    num_classes = y_train.shape[1]
    S_inputs = Input(shape=(None,), dtype='int32')
    embeddings = embedding_layer(S_inputs)
    # embeddings = Position_Embedding()(embeddings) # 增加Position_Embedding能轻微提高准确率
    O_seq = Attention(3, 32)([embeddings, embeddings, embeddings])
    O_seq = GlobalAveragePooling1D()(O_seq)
    O_seq = Dropout(0.5)(O_seq)
    outputs = Dense(num_classes, activation='sigmoid')(O_seq)
    model = Model(inputs=S_inputs, outputs=outputs)
    filepath="model/attention_"+model_name+"_{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, save_best_only=True,verbose=1, mode='auto')
    callbacks_list = [checkpoint]
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc',recall_threshold(0.5),precision_threshold(0.5)])
    model.fit(x_train, y_train,batch_size=128,epochs=30,validation_data=(x_test, y_test),callbacks=callbacks_list)
    score, acc,recall,precision= model.evaluate(x_test, y_test, batch_size=128)
    print("\nTest score: %.4f, accuracy: %.4f, recall: %.4f,precision: %.4f" % (score, acc,recall,precision))
    print('&&end&&' * 10)
