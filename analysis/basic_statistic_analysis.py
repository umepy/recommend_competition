#coding:utf-8
#データの基本統計量を調べる

import numpy as np
import pandas as pd
from numba import jit
import time
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool
import tqdm
from itertools import groupby
import pickle
import datetime

#データ読み込み
def read_data(name):
    train = pd.read_csv('../data/train/train_'+name+'.tsv',delimiter='\t',parse_dates=['time_stamp'])
    return train
def read_personal_data(name):
    with open('../data/personal/personal_'+name+'.pickle', 'rb') as f:
        data=pickle.load(f)
    return data

#基本統計量算出
def statistic_analysis(data,name=None):
    st=time.time()
    print(data.columns)
    print('number of uniqe ids: '+str(data.user_id.value_counts().count()))
    print('\nコンバージョン:3, クリック:2, 閲覧:1, カート:0')
    print(data['event_type'].value_counts())
    print('\n')
    print(data['ad'].value_counts())

    if name != None:
        #ユニークIDを取得
        personal_dic = read_personal_data('B')
        num=[]
        for i in personal_dic.keys():
            if len(personal_dic[i]) < 1000:
                num.append(len(personal_dic[i]))
        print('Time : '+str(time.time() - st))
        sns.distplot(num,kde=False,bins=100)
        print(num)
        plt.show()

#各個人のデータを抽出
def extract_personaldata(name,data):
    unique = data.user_id.unique()
    DataFrameDict = {elem: pd.DataFrame for elem in unique}
    for key in tqdm.tqdm(DataFrameDict.keys()):
        DataFrameDict[key] = data[:][data.user_id == key]
    with open('../data/personal/personal_'+name+'.pickle','wb') as f:
        pickle.dump(DataFrameDict,f)

#個人のデータをA,B,C,D全てで抽出
def extract_all():
    for i in ('A','B','C','D'):
        a=read_data(i)
        extract_personaldata(i,a)
        extract_ranking(i)
    get_predict_ids()

#各個人のテスト期間での商品上位とIDCGの算出
def extract_ranking(name):
    # 個人のデータ読み込み
    with open('../data/personal/personal_' + name + '.pickle', 'rb') as f:
        df = pickle.load(f)

    out_dic=dict()
    train_dic=dict()
    #各ユーザに対して商品の関連度とIDCGの辞書を作成し，user_idをキーとした辞書を作成
    for i in tqdm.tqdm(df.keys()):
        tmp_dic,tmp_train=extract_items(df[i])
        out_dic[i]=tmp_dic
        train_dic[i]=tmp_train

    with open('../data/personal/personal_test_items_IDCG_' + name + '.pickle','wb') as f:
        pickle.dump(out_dic,f)
    with open('../data/personal/personal_train_' + name + '.pickle','wb') as f:
        pickle.dump(train_dic,f)

#個人のデータから商品idと関連度を算出
def extract_items(data):
    test=data[data['time_stamp'] > datetime.datetime(year=2017, month=4, day=24)]
    train = data[data['time_stamp'] <= datetime.datetime(year=2017, month=4, day=24)]
    out_dic=dict()

    #テスト期間でのidの取得
    product_ids=pd.unique(test['product_id'])
    for id in product_ids:
        tmp_df=test[test['product_id']==id].reset_index()
        out_dic[id] = 0
        for j in range(len(tmp_df)):
            if out_dic[id] < tmp_df['event_type'][j] and tmp_df['ad'][j] != 0:
                out_dic[id] = tmp_df['event_type'][j]

    #IDCGの計算
    scores=list(out_dic.values())
    scores.reverse()
    out_dic['IDCG']=calc_IDCG(scores)
    return out_dic,train

#IDCGの計算
def calc_IDCG(rank):
    idcg=0
    if len(rank)>=22:
        for i in range(22):
            idcg+=(2**rank[i]-1)/np.log2((i+1)+1)
    else:
        for i in range(len(rank)):
            idcg += (2 ** rank[i] - 1) / np.log2((i + 1) + 1)
    return idcg

#予測用idの取得
def get_predict_ids():
    df=pd.read_csv('../data/sample_submit.tsv',delimiter='\t',names=['user_id','item_id','rank'])
    ids=pd.unique(df['user_id'])
    ids_dic={}
    tmp_A=[]
    tmp_B=[]
    tmp_C=[]
    tmp_D=[]
    for i in ids:
        if 'A' in i:
            tmp_A.append(i)
        elif 'B' in i:
            tmp_B.append(i)
        elif 'C' in i:
            tmp_C.append(i)
        else:
            tmp_D.append(i)
    ids_dic['A']=tmp_A
    ids_dic['B']=tmp_B
    ids_dic['C']=tmp_C
    ids_dic['D']=tmp_D

    with open('../data/submit_ids.pickle','wb') as f:
        pickle.dump(ids_dic,f)

if __name__=='__main__':
    #a=read_data('D')
    #statistic_analysis(a,'B')
    #extract_personaldata('D',a)
    get_predict_ids()