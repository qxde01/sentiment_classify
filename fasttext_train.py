#! -*- coding: utf-8 -*-
import  fasttext
vector_size=128
word_ngrams=3
words_file='data/comments_w2v_train.txt'
cbow_file='model/word2vec_cbow_'+str(vector_size)
w2v_model = fasttext.cbow(words_file, cbow_file, lr=0.1,dim=vector_size,min_count=2,word_ngrams=word_ngrams,minn=1,maxn=15,ws=3,silent=0,epoch=50,bucket=200000)
print("&"*50)

skip_file='model/word2vec_skip_'+str(vector_size)
w2v_model = fasttext.skipgram(words_file, skip_file, lr=0.1,dim=vector_size,min_count=2,word_ngrams=word_ngrams,minn=1,maxn=15,ws=3,silent=0,epoch=50,bucket=200000)
print("&"*50)

classify5_file="model/fasttext_sentiment_5c_"+str(vector_size)
classify5_model=fasttext.supervised('data/comments_fasttext_train.txt', classify5_file, label_prefix="__label__", dim=vector_size, min_count=2, ws=3,word_ngrams=word_ngrams, minn=1, maxn=15, epoch=20, silent=0, bucket=200000)
test5 = classify5_model.test('data/comments_fasttext_test.txt')
train5_pred=classify5_model.test('data/comments_fasttext_train.txt')
print('train data ：precision=%f,recall=%f'%(train5_pred.precision,train5_pred.recall))
print('test data ：precision=%f,recall=%f'%(test5.precision,test5.recall))

classify2_file="model/fasttext_sentiment_2c_"+str(vector_size)
classify2_model=fasttext.supervised('data/comments_fasttext_train_2c.txt', classify2_file, label_prefix="__label__", dim=vector_size, min_count=2, ws=3,word_ngrams=word_ngrams, minn=1, maxn=15, epoch=20, silent=0, bucket=200000)
test2 = classify2_model.test('data/comments_fasttext_test_2c.txt')
train2_pred=classify2_model.test('data/comments_fasttext_train_2c.txt')
print('train data ：precision=%f,recall=%f'%(train2_pred.precision,train2_pred.recall))
print('test data ：precision=%f,recall=%f'%(test2.precision,test2.recall))