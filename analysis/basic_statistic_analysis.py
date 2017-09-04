#coding:utf-8
#データの基本統計量を調べる

import numpy as np
import pandas as pd

#データ読み込み
def read_data():
    trainA = pd.read_csv('../data/train/train_A.tsv',delimiter='\t')
    trainB = pd.read_csv('../data/train/train_B.tsv', delimiter='\t')
    trainC = pd.read_csv('../data/train/train_C.tsv', delimiter='\t')
    trainD = pd.read_csv('../data/train/train_D.tsv', delimiter='\t')
    return trainA,trainB,trainC,trainD

#基本統計量算出
@profile
def statistic_analysis(data):
    print('number of uniqe ids: '+str(data.user_id.value_counts().count()))

if __name__=='__main__':
    a,b,c,d=read_data()
    statistic_analysis(a)