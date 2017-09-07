#coding:utf-8
#交差検定環境の構築
#4月の30日中、24日学習して6日テストとする

import pandas as pd
import numpy as np
from numba import jit
import datetime
import pickle
import random

class CrossValidation():
    def __init__(self,name,K=5):
        self.name=name
        self.K = K
        self.read_data()
        self.split_data()

    #データを読み込み分割
    def read_data(self):
        #個人のデータ読み込み
        with open('../data/personal/personal_test_items_IDCG_'+self.name+'.pickle','rb') as f:
            self.personal_result=pickle.load(f)
        with open('../data/personal/personal_train_' + self.name + '.pickle', 'rb') as f:
            self.personal_train=pickle.load(f)

    #データを分割
    def split_data(self):
        ran_ids=list(self.personal_train.keys())
        random.shuffle(ran_ids)
        self.cv_trains=[]
        self.cv_tests=[]
        size=int(len(ran_ids)/self.K)
        for i in range(self.K):
            tmp_test=ran_ids[i*size:(i+1)*size]
            tmp_train=list(set(ran_ids)-set(tmp_test))
            self.cv_trains.append(tmp_train)
            self.cv_tests.append(tmp_test)

    #誤差関数
    def DCG(self,user_id,items):
        #まずDCGを計算
        DCG=0
        for i in range(len(items)):
            if i in self.personal_result[user_id].keys():
                DCG+=(2**self.personal_result[user_id][i]-1)/np.log2(i+2)
        return DCG/self.personal_result[user_id]['IDCG']


if __name__=='__main__':
    a=CrossValidation('B')