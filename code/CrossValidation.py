#coding:utf-8
#交差検定環境の構築
#4月の30日中、24日学習して6日テストとする

import pandas as pd
import numpy as np
from numba import jit
import datetime
import pickle

class CrossValidation():
    def __init__(self,name):
        self.name=name
        self.read_data()

    #データを読み込み分割
    def read_data(self):
        data=pd.read_csv('../data/train/train_'+self.name+'.tsv',delimiter='\t',parse_dates=['time_stamp'])
        #データを学習用，テスト用に分割
        self.test=data[data.time_stamp > datetime.datetime(year=2017,month=4,day=24)]
        self.train=data[data.time_stamp <= datetime.datetime(year=2017, month=4, day=24)]

        #個人のデータ読み込み
        with open('../data/personal/personal_'+self.name+'.pickle','rb') as f:
            df=pickle.load(f)
            print(df['0000000_B'][df['0000000_B']['time_stamp']>datetime.datetime(year=2017,month=4,day=24)])

    #誤差関数
    def DCG(self,user,item):
        pass


if __name__=='__main__':
    a=CrossValidation('B')