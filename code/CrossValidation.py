#coding:utf-8
#交差検定環境の構築
#4月の30日中、24日学習して6日テストとする

import pandas as pd
import numpy as np
from numba import jit
import datetime
import pickle
import random
import time


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
        #IDCGが0の場合の分岐
        if self.personal_result[user_id]['IDCG']==0:
            return -1
        #まずDCGを計算
        DCG=0
        for i in range(len(items)):
            if items[i] in list(self.personal_result[user_id].keys()):
                DCG+=(2**self.personal_result[user_id][items[i]]-1)/np.log2(i+2)
        return DCG/self.personal_result[user_id]['IDCG']

    #評価関数
    def evaluate(self,predict):
        score=0.0
        count=0
        for i in predict.keys():
            tmp=self.DCG(i,predict[i])
            if tmp==-1:
                count+=1
            else:
                score+=tmp
        return score/count

    #方法1 - 過去のユーザの行動履歴から推薦(評価の高いもの順)
    def method_choice_from_past_data(self,test_ids):
        predict_test={}
        for i in test_ids:
            #ユニークitem idを取得
            past_items=pd.unique(self.personal_train[i]['product_id'])
            random.shuffle(past_items)
            if len(past_items) > 22:
                past_items=past_items[:22]
            predict_test[i]=past_items
        return self.evaluate(predict_test)

    #Cross-validationの実行
    def CV(self):
        print('CV開始いたします')
        score_sum=0
        for i in range(self.K):
            score_tmp=self.method_choice_from_past_data(self.cv_tests[i])
            print(str(i)+' of '+str(self.K)+' CrossValidation, score :'+str(score_tmp))
            score_sum+=score_tmp
        print('Final Score '+ self.name +' : '+str(score_sum/self.K))
        return score_sum/self.K

def all_CV(number=5):
    scores={'A':0,'B':0,'C':0,'D':0}
    for _ in range(number):
        for i in ['A','B','C','D']:
            a=CrossValidation(i)
            scores[i]+=a.CV()
    print(str(number) + '回平均結果')
    for i in ['A', 'B', 'C', 'D']:
        scores[i]/=number
        print(i + '\t' + str(scores[i]))

if __name__=='__main__':
    all_CV()