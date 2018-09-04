# -*- coding: utf-8 -*-
from hanziconv import HanziConv
from joblib import Parallel, delayed
import pandas as pd
from collections import Counter
import re
import  jieba
import random
def load_user_dict(dictfile='data/dict.csv'):
    dict_words=pd.read_csv(dictfile)
    dict_words=dict_words[dict_words.freq>9]
    n=dict_words.shape[0]
    for i in range(0,n):
        try:
            jieba.add_word(dict_words.char[i],int(dict_words.len[i])*1000,'nz')
        except Exception as e:
            print(e)

stopwords=open('data/stopwords.txt').readlines()
stopwords=[w.strip() for w in  stopwords]
def words_seg(x):
    w = jieba.cut(x.upper())
    w = [ word.strip() for word in w if word not in stopwords]
    w=[word.strip() for word in w if len(word)>0]
    return [' '.join(w), len(w)]

def seg(k):
    words,words_size=words_seg(X.content[k])
    return X.shop_url[k],X.post_time[k],X.content[k],X.score[k],words,words_size


X=pd.read_csv('data/comments_raw_v3.csv')
'''
shop_url=X.shop_url.tolist()
deal_url=X.deal_url.tolist()
mn=len(shop_url)
shop=[]
for i in range(0,mn):
    if len(str(shop_url[i]))<6:
        shop.append(deal_url[i])
    else:
        shop.append(shop_url[i])
X=X.loc[:,['post_time', 'content', 'score']]
X.insert(0,'shop_url',shop)
'''
def to_S(k):
    txt=X.content[k].strip()
    txt = re.sub('\t|\r', '\n',txt)
    txt = txt.replace('\n\n', '\n')
    txt = re.sub('  |\u3000', ' ', txt)
    txt=HanziConv.toSimplified(txt)
    txt=txt.strip()
    return (X.shop_url[k], X.post_time[k],txt,int(X.score[k]),len(txt))
n=X.shape[0]
'''
X1= Parallel(n_jobs=18, verbose=1, pre_dispatch='2*n_jobs')(delayed(to_S)(k) for k in range(0,n))
X1=[x[:4] for x in X1 if x[4]>0]
X1=pd.DataFrame(X1,columns=['shop_url', 'post_time', 'content', 'score'])
X=X1.sort_values(by="score")
'''
load_user_dict(dictfile='data/dict_utf8.csv')
n=X.shape[0]
X2= Parallel(n_jobs=16, verbose=1, pre_dispatch='2*n_jobs')(delayed(seg)(k) for k in range(0,n))
X2=[x for x in X2 if x[5]>0]
X2=pd.DataFrame(X2,columns=['shop_url', 'post_time', 'content', 'score', 'words','words_size'])
X2=X2.sort_values(by="score")
ids=[i+1 for i in range(0,X2.shape[0])]
X2.insert(0,'id',ids)

X2.to_csv('data/comments_words_v4.csv',index=False,encoding="UTF-8")
X3=X2.loc[:,['id','score','words','words_size']]
X3.to_csv('data/comments_words_std_v4.csv',index=False,encoding="UTF-8")
X4=X3[X3.words_size>11]
X4=X4[X4.words_size<141]
Counter(X4.score.tolist())
Counter(X4.words_size.tolist())
X4.to_csv('data/comments_words_clean_std_v4.csv',index=False,encoding="UTF-8")
#### 句子切分，作为训练词向量语料
sentences= '\n'.join(X.content.tolist())
sentences=re.split('[\n\t\r。]',sentences)
sentences=[s for s in sentences if len(s)>0]
sentences=list(set(sentences))

sentences_words= Parallel(n_jobs=20, verbose=1, pre_dispatch='2*n_jobs')(delayed(words_seg)(k) for k in sentences)
sentences_words=[x[0] for x in sentences_words if x[1]>0]
sentences_words1=[x+'\n' for x in sentences_words  ]
fo = open("data/comments_w2v_train_words_v4.txt", "w")
fo.writelines( sentences_words1 )
fo.close()

#################################################################
def gen_sample(datafile='data/comments_words_v3.csv',
               simfile='data/comments_words_std_v3_sim.csv',
               outfile='data/comments_v3'):
    A1=pd.read_csv(datafile)
    #words_char=A1.words.tolist()
    #words_char_num=[len(x.replace(' ','')) for x in words_char]
    #A1.insert(7, 'words_char_size', words_char_num)
    A2=pd.read_csv(simfile)
    A3= A1[A1.id.isin(A2.id) ]
    A4=pd.merge(A3,A2,on='id',how='left')
    A4=A4.drop('score_y',axis=1)
    A4.columns=['id', 'shop_url', 'post_time', 'content', 'score', 'words', 'words_size', 'sim_id', 'sim_score', 'sim_len']
    #A4.columns = ['id', 'score', 'words', 'words_size', 'sim_id', 'sim_score', 'sim_len']
    #A4=A4[A4.words_size>15]
    print('data size:',A4.shape[0])
    A4[A4.sim_len>1].to_csv(outfile+'_sim_view.csv',index=False,encoding='UTF-8')
    #A5 = A4[A4.id == A4.sim_id]
    #print('del sim data :', A5.shape[0])
    A6=A4[A4.sim_len==1]
    print('no sim data :', A6.shape[0])
    A6=A6.loc[:,['id','score','words']]
    scores_stat=Counter(A6.score.tolist())
    print('score dis:',scores_stat)
    scores_stat=[(s1,s2) for s1,s2 in scores_stat.items()]
    scores_stat=sorted(scores_stat, key=lambda x: x[1],reverse=False)
    n_sample=scores_stat[0][1]
    #random.sample(range(0,),n_sample)
    Z1=A6[A6.score==1]
    #Z1=Z1.iloc[sorted(random.sample(range(0,Z1.shape[0]),n_sample))]
    Z1=Z1.sample(frac=n_sample/Z1.shape[0])
    Z2=A6[A6.score==2]
    Z2=Z2.sample(frac=n_sample/Z2.shape[0])
    Z3=A6[A6.score==3]
    Z3=Z3.sample(frac=n_sample/Z3.shape[0])
    Z4=A6[A6.score==4]
    Z4=Z4.sample(frac=n_sample/Z4.shape[0])
    Z5=A6[A6.score==5]
    Z5=Z5.sample(frac=n_sample/Z5.shape[0])
    Z=pd.concat([Z1,Z2,Z3,Z4,Z5])
    Z=Z.sample(frac=1).reset_index(drop=True)
    print('sample data :', Z.shape[0])
    Z.to_csv(outfile+'_words_sample.csv',index=False,encoding='UTF-8')
    #Z_train=Z.sample(frac=0.8).reset_index(drop=True)
    #Z_train.to_csv(outfile+'_train_sample.csv',index=False,encoding='UTF-8')
    #Z_test=Z[~Z.id.isin(Z_train.id)]
    #Z_test.to_csv(outfile+'_test_sample.csv',index=False,encoding='UTF-8')

gen_sample(datafile='data/comments_words_v5_L4_7.csv',simfile='data/comments_words_sim_v3_32_0.85_L4-7.csv',outfile='data/comments_v5_32_0.60_L4-7')
gen_sample(datafile='data/comments_words_v5.csv',simfile='data/comments_words_std_v4_64_0.60_L4-10.csv',outfile='data/comments_v4_64_0.60_L4-10')

A1=pd.read_csv('data/comments_words_v4.csv')
s=Counter(A1.words)
s1=[k for k,v in s.items() if v==1]
A2= A1[A1.words.isin(s1) ]
A2.to_csv('data/comments_words_v5.csv',index=False,encoding="UTF-8")
A3=A2.loc[:,['id','score','words','words_size']]
n1=51;n2=120
A31=A3[A3.words_size>=n1]
A32=A31[A31.words_size<=n2]
A32.to_csv('data/comments_words_v5_L'+str(n1)+'_'+str(n2)+'.csv',index=False,encoding="UTF-8")

'''
content=X1.content.tolist()
char=list(' '.join(content))
fq=Counter(char)
freq=[(x[0],x[1]) for x in fq.items()]
freq=pd.DataFrame(freq,columns=['char','freq'])
freq=freq.sort_values(by="freq",ascending=False)
freq.to_csv('data/freq.csv',index=False)
'''
# sp="[，。！#….～\\?,【】!\\^\\)\\(；）（-_：⊙”“/\\\\\[\\]\\+￣′·`;@=．\\|>‘<°｀」》《’丶﹏˙￥&ノ＿\\$ \n\t]"
'''
## 统计常用词语，作为扩展词库
def replace_char(x):
    sp="[˵⁻♡♀_◎ω＼↗↖｢」》《•๑⊙′/╯▽╰～ಡ·•●з」∠ε`~。，；：“”‘’？【】#￥（）、！,!':;\\|\\-\\*\\\…\\.\\[\\]\\(\\)\\?\n\r\t ]|[\uD800-\uDBFF][\uDC00-\uDFFF]|[0-9A-Za-z]|\u3000"
    x=re.sub(sp,' ',x)
    return x.strip()

txt= Parallel(n_jobs=18, verbose=1, pre_dispatch='2*n_jobs')(delayed(replace_char)(k) for k in X1.content)
txt=' '.join(txt)
txt=txt.split(' ')
sentences=Counter(txt)
sentences=[(x[0],x[1],len(x[0])) for x in sentences.items()]
sentences=pd.DataFrame(sentences,columns=['char','freq','len'])
sentences=sentences.sort_values(by="freq",ascending=False)
sentences_1=sentences[sentences.freq>50 ]
sentences_1=sentences_1[ sentences_1.len<6]
sentences_1=sentences_1[ sentences_1.len>1]
sentences_1.to_csv('data/sentences.csv',index=False,encoding="GBK")

'''
#import thulac

#thu1 = thulac.thulac(user_dict="data/thu_dict.txt", model_path=None, T2S=True, seg_only=True, filt=True, deli='___')