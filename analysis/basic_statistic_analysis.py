#coding:utf-8
#データの基本統計量を調べる

import numpy as np
import pandas as pd
from numba import jit
import time
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool

#データ読み込み
def read_data():
    trainA = pd.read_csv('../data/train/train_A.tsv',delimiter='\t')
    trainB = pd.read_csv('../data/train/train_B.tsv', delimiter='\t')
    trainC = pd.read_csv('../data/train/train_C.tsv', delimiter='\t')
    trainD = pd.read_csv('../data/train/train_D.tsv', delimiter='\t')
    return trainA,trainB,trainC,trainD

#基本統計量算出
#@jit
def statistic_analysis(data):
    st=time.time()
    print(data.columns)
    print('number of uniqe ids: '+str(data.user_id.value_counts().count()))
    print('\nコンバージョン:3, クリック:2, 閲覧:1, カート:0')
    print(data['event_type'].value_counts())
    print('\n')
    print(data['ad'].value_counts())

    #ユニークIDを取得
    unique=data.user_id.unique()
    num = []
    for i in unique:
        tmp=data[data['user_id'].isin([i])]
        num.append(len(tmp))
    print('Time : '+str(time.time() - st))
    sns.distplot(num)
    plt.show()

if __name__=='__main__':
    a,b,c,d=read_data()
    statistic_analysis(a)