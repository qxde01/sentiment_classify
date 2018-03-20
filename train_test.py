#! -*- coding: utf-8 -*-
from preprocess import *
from model_train import *

data_file = 'data/comments_words_clean_sample.csv'
word2vec_file = 'model/word2vec_skip_128.bin'

max_num_words = 70000
max_text_word = 75
x_train, y_train, x_test, y_test, embedding_matrix, embedding_layer = build_data(
    data_file=data_file, word2vec_file=word2vec_file, max_num_words=max_num_words,
    max_text_word=max_text_word, classes=[1, 5], vector=False)
print('train biLSTM model of score 1,5 .....')
biLSTM_train(x_train, y_train, x_test, y_test, input_shape=(max_text_word, 128), embedding_layer=embedding_layer,
             model_name='score_1_5')
print('train CNN1D model of score 1,5 .....')
CNN1D_train(x_train, y_train, x_test, y_test, max_text_word=max_text_word, embedding_layer=embedding_layer,
            model_name='score_1_5')
print('train LSTM model of score 1,5 .....')
LSTM_train(x_train, y_train, x_test, y_test, embedding_layer=embedding_layer, model_name='score_1_5')
print('train CNN_LSTM model of score 1,5 .....')
CNN_LSTM_train(x_train, y_train, x_test, y_test, embedding_layer=embedding_layer, model_name='score_1_5')
print('train GRU_Capsule model of score 1,5 .....')
GRU_Capsule_train(x_train, y_train, x_test, y_test, gru_len=128, embedding_layer=embedding_layer,
                  model_name='score_1_5')

x_train, y_train, x_test, y_test, embedding_matrix, embedding_layer = build_data(
    data_file=data_file, word2vec_file=word2vec_file, max_num_words=max_num_words,
    max_text_word=max_text_word, classes=[1, 3, 5], vector=False)
print('train biLSTM model of score 1,3,5 .....')
biLSTM_train(x_train, y_train, x_test, y_test, input_shape=(max_text_word, 128), embedding_layer=embedding_layer,
             model_name='score_1_3_5')
print('train CNN1D model of score 1,3,5 .....')
CNN1D_train(x_train, y_train, x_test, y_test, max_text_word=max_text_word, embedding_layer=embedding_layer,
            model_name='score_1_3_5')
print('train LSTM model of score 1,3,5 .....')
LSTM_train(x_train, y_train, x_test, y_test, embedding_layer=embedding_layer, model_name='score_1_3_5')
print('train CNN_LSTM model of score 1,3,5 .....')
CNN_LSTM_train(x_train, y_train, x_test, y_test, embedding_layer=embedding_layer, model_name='score_1_3_5')
print('train GRU_Capsule model of score 1,3,5 .....')
GRU_Capsule_train(x_train, y_train, x_test, y_test, gru_len=128, embedding_layer=embedding_layer,
                  model_name='score_1_3_5')

x_train, y_train, x_test, y_test, embedding_matrix, embedding_layer = build_data(
    data_file=data_file, word2vec_file=word2vec_file, max_num_words=max_num_words,
    max_text_word=max_text_word, classes=[2, 4], vector=False)
print('train biLSTM model of score 2,4 .....')
biLSTM_train(x_train, y_train, x_test, y_test, input_shape=(max_text_word, 128), embedding_layer=embedding_layer,
             model_name='score_2_4')
print('train CNN1D model of score 2,4 .....')
CNN1D_train(x_train, y_train, x_test, y_test, max_text_word=max_text_word, embedding_layer=embedding_layer,
            model_name='score_2_4')
print('train LSTM model of score 2,4 .....')
LSTM_train(x_train, y_train, x_test, y_test, embedding_layer=embedding_layer, model_name='score_2_4')
print('train CNN_LSTM model of score 2,4 .....')
CNN_LSTM_train(x_train, y_train, x_test, y_test, embedding_layer=embedding_layer, model_name='score_2_4')
print('train GRU_Capsule model of score 2,4 .....')
GRU_Capsule_train(x_train, y_train, x_test, y_test, gru_len=128, embedding_layer=embedding_layer,
                  model_name='score_2_4')

x_train, y_train, x_test, y_test, embedding_matrix, embedding_layer = build_data(
    data_file=data_file, word2vec_file=word2vec_file, max_num_words=max_num_words,
    max_text_word=max_text_word, classes=[1, 2, 3, 4, 5], vector=False)
print('train biLSTM model of score 1,2,3,4,5 .....')
biLSTM_train(x_train, y_train, x_test, y_test, input_shape=(max_text_word, 128), embedding_layer=embedding_layer,
             model_name='score_1_2_3_4_5')
print('train CNN1D model of score 1,2,3,4,5 .....')
CNN1D_train(x_train, y_train, x_test, y_test, max_text_word=max_text_word, embedding_layer=embedding_layer,
            model_name='score_1_2_3_4_5')
print('train LSTM model of score 1,2,3,4,5 .....')
LSTM_train(x_train, y_train, x_test, y_test, embedding_layer=embedding_layer, model_name='score_1_2_3_4_5')
print('train CNN_LSTM model of score 1,2,3,4,5 .....')
CNN_LSTM_train(x_train, y_train, x_test, y_test, embedding_layer=embedding_layer, model_name='score_1_2_3_4_5')
print('train GRU_Capsule model of score 1,2,3,4,5 .....')
GRU_Capsule_train(x_train, y_train, x_test, y_test, gru_len=128, embedding_layer=embedding_layer,
                  model_name='score_1_2_3_4_5')

