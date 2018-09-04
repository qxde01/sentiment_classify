# coding:utf-8
from datasketch import MinHash, MinHashLSH
import pickle,csv,time
import pandas as pd
from tqdm import tqdm
from collections import Counter

def text2lsh(filename='data/comments_words_std.csv',threshold=0.9,num_perm=128,is_save=True,lshfile='data/output'):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    X = pd.read_csv(filename)
    #X=X[X.words_size>3]
    n=X.shape[0]
    hash_list=[]
    for i in tqdm(range(0,n)):
        try:
            id=int(X.id[i])
            score=int(X.score[i])
            m = MinHash(num_perm=num_perm)
            words=X.words[i].split(' ')
            for d in words:
                m.update(d.encode('utf8'))
            lsh.insert((id, score), m)
            hash_list.append((id,m))
        except Exception as e:
            print(e,words)
            #print(id, score)
    if is_save:
        f = open(lshfile+'_lsh.pkl', 'wb')
        pickle.dump(lsh, f, 0)
        f.close()
        f = open(lshfile+'_hash_list.pkl', 'wb')
        pickle.dump(hash_list, f, 0)
        f.close()
    return lsh,hash_list

def lshsim(lsh,hash_list,is_save=True,outfile='output.csv'):
    has_ids=set()
    out=[]
    global lsh_var
    lsh_var=lsh
    n=len(hash_list)
    for i in tqdm(range(0,n)):
        try:
            id,mh=hash_list[i]
            if has_ids.__contains__(id) == False:
                resu = lsh_var.query(mh)
                num = len(resu)
                s=Counter([x[1] for x in resu])
                s=[(sc ,sm) for sc,sm in s.items()]
                s=sorted(s, key=lambda x: x[1],reverse=True)
                score=s[0][0]
                tmp_resu = [(x[0], x[1], id, score, num) for x in resu]
                tmp_id = [int(x[0]) for x in resu]
                has_ids = has_ids.union(set(tmp_id))
                out = out + tmp_resu
                for x in resu:
                    try:
                       lsh_var.remove((int(x[0]), int(x[1])))
                    except Exception as e:
                       print(e,x)
        except Exception as e:
            print(i,e)
    out=pd.DataFrame(out,columns=["id", "score", "sim_id","sim_score",'sim_len'])
    if is_save :
        out.to_csv(outfile,index=False,encoding='UTF-8')
    return out

#lsh=text2lsh(filename='data/comments_words_std.csv')
if __name__ == "__main__":
    lsh,hash_list = text2lsh(filename='data/comments_words_v5_L8_11.csv',threshold=0.6,num_perm=32,lshfile='data/comments_v5_32_0.60_L8-11')
    output = lshsim(lsh=lsh, hash_list=hash_list, outfile='data/comments_words_sim_v3_32_0.85_L8-11.csv')
    #f = open('data/comments_v2_lsh.pkl', 'rb')
    #lsh = pickle.load(f)
    #f.close()

