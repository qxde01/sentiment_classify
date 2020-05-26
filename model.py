from tensorflow import keras
#import keras
def con1d_bn(x,filters=128,k=3):
    x=keras.layers.Conv1D(filters=filters,kernel_size=k,activation=None,padding='same')(x)
    x=keras.layers.BatchNormalization(momentum=0.95)(x)
    x=keras.layers.PReLU()(x)
    print(x)
    return x

def TextCNN(weights,num_words=3344,max_length=200,classes=3):
    #embedding_layer = Embedding(num_words, embeddings_dim, weights=[embedding_matrix], input_length=max_text_word, trainable=False)
    inputs=keras.layers.Input(shape=(max_length,),name='input')
    x=keras.layers.Embedding(output_dim=128,input_dim=num_words,input_length=max_length,name='embedding',weights=[weights],trainable=True)(inputs)
    #x1 = keras.layers.SpatialDropout1D(0.45)(x)
    x1=con1d_bn(x, filters=128, k=3)
    #x1 = con1d_bn(x1, filters=256, k=1)
    #x2 = keras.layers.SpatialDropout1D(0.45)(x)
    x2=con1d_bn(x, filters=128, k=5)
    x2 = con1d_bn(x2, filters=256, k=3)
    #x3 = keras.layers.SpatialDropout1D(0.45)(x)
    x3=con1d_bn(x, filters=128, k=7)
    #x3 = con1d_bn(x3, filters=256, k=5)
    x3 = con1d_bn(x3, filters=512, k=3)
    x1=keras.layers.AveragePooling1D(pool_size=3+max_length-4)(x1)
    print(x1)
    x2= keras.layers.AveragePooling1D(pool_size=3+max_length-4)(x2)
    print(x2)
    x3 = keras.layers.AveragePooling1D(pool_size=3+max_length-4)(x3)
    print(x3)

    output=keras.layers.concatenate([x1,x2,x3])
    output=keras.layers.Flatten()(output)
    #output=keras.layers.GlobalAveragePooling1D()(output)
    output=keras.layers.Dropout(0.55)(output)
    output=keras.layers.Dense(classes,activation='softmax')(output)
    model=keras.models.Model(inputs=inputs,outputs=output)
    return model

def build_biLSTM(weights,num_words=3344,max_length=200,classes=3):
    inputs=keras.layers.Input(shape=(max_length,),name='input')
    x = keras.layers.Embedding(output_dim=128, input_dim=num_words, input_length=max_length, name='embedding', weights=[weights], trainable=True)(inputs)
    #x=keras.layers.Embedding(output_dim=256,input_dim=num_words,input_length=max_length,name='embedding')(inputs)
    x=keras.layers.SpatialDropout1D(0.35)(x)
    x1=keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True), input_shape=(max_length, 100))(x)
    x1=keras.layers.Bidirectional(keras.layers.LSTM(256))(x1)
    #x1 = keras.layers.Bidirectional(keras.layers.GRU(128, return_sequences=True), input_shape=(max_length, 100))(x)
    #x1 = keras.layers.Bidirectional(keras.layers.GRU(256))(x1)

    #output=keras.layers.Flatten()(x1)
    #output=keras.layers.GlobalAveragePooling1D()(output)
    #output=keras.layers.Dropout(0.6)(x1)
    output=keras.layers.Dense(classes,activation='softmax')(x1)
    model=keras.models.Model(inputs=inputs,outputs=output)
    return model

def DPCNN(weights,num_words=3344,max_length=200,classes=3):
    #x0 = Input(shape=(maxlen,))
    filter_nr = 64
    filter_size = 3
    max_pool_size = 3
    max_pool_strides = 2
    dense_nr = 512
    spatial_dropout = 0.35
    dense_dropout = 0.5
    inputs=keras.layers.Input(shape=(max_length,),name='input')
    x = keras.layers.Embedding(output_dim=100, input_dim=num_words, input_length=max_length, name='embedding', weights=[weights], trainable=False)(inputs)

    #emb_comment = Embedding(max_features, embed_size)(x0)
    x = keras.layers.SpatialDropout1D(spatial_dropout)(x)
    block1 =  keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(x)
    block1 =  keras.layers.BatchNormalization()(block1)
    block1 = keras.layers.PReLU()(block1)
    block1 =  keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block1)
    block1 =  keras.layers.BatchNormalization()(block1)
    block1 = keras.layers. PReLU()(block1)
    # we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
    # if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add emb_comment directly to block1_output
    resize_emb =  keras.layers.Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear')(x)
    resize_emb =  keras.layers.PReLU()(resize_emb)
    block1_output =  keras.layers.add([block1, resize_emb])
    block1_output =  keras.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)
    block2 =  keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block1_output)
    block2 =  keras.layers.BatchNormalization()(block2)
    block2 = keras.layers. PReLU()(block2)
    block2 =  keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block2)
    block2 =  keras.layers.BatchNormalization()(block2)
    block2 =  keras.layers.PReLU()(block2)
    block2_output =  keras.layers.add([block2, block1_output])
    block2_output = keras.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)
    block3 =  keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block2_output)
    block3 =  keras.layers.BatchNormalization()(block3)
    block3 =  keras.layers.PReLU()(block3)
    block3 =  keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block3)
    block3 =  keras.layers.BatchNormalization()(block3)
    block3 =  keras.layers.PReLU()(block3)
    block3_output =  keras.layers.add([block3, block2_output])
    block3_output =  keras.layers.MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)
    block4 =  keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block3_output)
    block4 =  keras.layers.BatchNormalization()(block4)
    block4 =  keras.layers.PReLU()(block4)
    block4 =  keras.layers.Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block4)
    block4 =  keras.layers.BatchNormalization()(block4)
    block4 =  keras.layers.PReLU()(block4)
    output =  keras.layers.add([block4, block3_output])
    output =  keras.layers.GlobalMaxPooling1D()(output)
    output =  keras.layers.Dense(dense_nr, activation='linear')(output)
    output =  keras.layers.BatchNormalization()(output)
    output = keras.layers.PReLU()(output)
    output =keras.layers.Dropout(dense_dropout)(output)
    #output = Dense(1, activation='sigmoid')(output)
    output = keras.layers.Dense(classes, activation='softmax')(output)
    model=keras.models.Model(inputs=inputs,outputs=output)
    return model
