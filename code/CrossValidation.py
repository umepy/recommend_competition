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
from multiprocessing import Process,Pool,Manager
from operator import itemgetter
from sklearn.decomposition import NMF


class CrossValidation():
    def __init__(self,name,K=5,method=None):
        self.name=name
        self.K = K
        self.read_data()
        self.split_data()
        self.method_func=self.choice_func(method)

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

    # 方法1 - 過去のユーザの行動履歴から推薦(ランダム抽出推薦)
    def method1_random_choice(self,num):
        test_ids=self.cv_tests[num]
        predict_test={}
        for i in test_ids:
            #ユニークitem idを取得
            past_items=pd.unique(self.personal_train[i]['product_id'])
            random.shuffle(past_items)
            if len(past_items) > 22:
                past_items=past_items[:22]
            predict_test[i]=past_items
        return self.evaluate(predict_test)

    # 方法2 - 過去のユーザの行動履歴から推薦(簡単な評価からの順位付け推薦)
    def method2_ranked_choice(self, num):
        test_ids = self.cv_tests[num]
        predict_test = {}
        for i in test_ids:
            # ユニークitem idを取得
            tmp_dict={}
            past_items = pd.unique(self.personal_train[i]['product_id'])

            #過去のデータから商品の重みを計算
            for j in past_items:
                tmp_dict[j]=0
                for k in self.personal_train[i][self.personal_train[i]['product_id']==j]['event_type']:
                    if tmp_dict[j]<k:
                        tmp_dict[j]=k
            sorted_list = sorted(tmp_dict.items(), key=itemgetter(1), reverse=True)
            if len(sorted_list) > 22:
                sorted_list = sorted_list[:22]
            predict_test[i] = [ x for x,y in sorted_list]
        return self.evaluate(predict_test)

    # 方法3 - 過去のユーザの行動履歴から推薦(購買商品は推薦しない)
    def method3_ranked_conversion_ignore_choice(self, num):
        test_ids = self.cv_tests[num]
        predict_test = {}
        for i in test_ids:
            # ユニークitem idを取得
            tmp_dict = {}
            past_items = pd.unique(self.personal_train[i]['product_id'])

            # 過去のデータから商品の重みを計算
            for j in past_items:
                tmp_dict[j] = 0
                for k in self.personal_train[i][self.personal_train[i]['product_id'] == j]['event_type']:
                    if tmp_dict[j] < k:
                        tmp_dict[j] = k
                    if k==3:
                        del tmp_dict[j]
                        break
            sorted_list = sorted(tmp_dict.items(), key=itemgetter(1), reverse=True)
            if len(sorted_list) > 22:
                sorted_list = sorted_list[:22]
            predict_test[i] = [x for x, y in sorted_list]
        return self.evaluate(predict_test)

    # 方法4 - カート商品を重視した順位付け
    def method4_ranked_cart(self, num):
        test_ids = self.cv_tests[num]
        predict_test = {}
        for i in test_ids:
            # ユニークitem idを取得
            tmp_dict = {}
            past_items = pd.unique(self.personal_train[i]['product_id'])

            # 過去のデータから商品の重みを計算
            for j in past_items:
                tmp_dict[j] = 0
                for k in self.personal_train[i][self.personal_train[i]['product_id'] == j]['event_type']:
                    if k == 0:
                        tmp_dict[j] +=1
                    if k==-1:
                        del tmp_dict[j]
                        break
            sorted_list = sorted(tmp_dict.items(), key=itemgetter(1),reverse=True)
            if len(sorted_list) > 22:
                sorted_list = sorted_list[:22]
            predict_test[i] = [x for x, y in sorted_list]
        return self.evaluate(predict_test)

    # 方法5 - クリックした商品を重視した順位付け
    def method5_ranked_click(self, num):
        test_ids = self.cv_tests[num]
        predict_test = {}
        for i in test_ids:
            # ユニークitem idを取得
            tmp_dict = {}
            past_items = pd.unique(self.personal_train[i]['product_id'])

            # 過去のデータから商品の重みを計算
            for j in past_items:
                tmp_dict[j] = 0
                for k in self.personal_train[i][self.personal_train[i]['product_id'] == j]['event_type']:
                    if k == 2:
                        tmp_dict[j] += 1
                    if k == -1:
                        del tmp_dict[j]
                        break
            sorted_list = sorted(tmp_dict.items(), key=itemgetter(1), reverse=True)
            if len(sorted_list) > 22:
                sorted_list = sorted_list[:22]
            predict_test[i] = [x for x, y in sorted_list]
        return self.evaluate(predict_test)

    # 方法6 - 閲覧した商品を重視した順位付け
    def method6_ranked_view(self, num):
        test_ids = self.cv_tests[num]
        predict_test = {}
        for i in test_ids:
            # ユニークitem idを取得
            tmp_dict = {}
            past_items = pd.unique(self.personal_train[i]['product_id'])

            # 過去のデータから商品の重みを計算
            for j in past_items:
                tmp_dict[j] = 0
                for k in self.personal_train[i][self.personal_train[i]['product_id'] == j]['event_type']:
                    if k == 1:
                        tmp_dict[j] += 1
                    if k == -1:
                        del tmp_dict[j]
                        break
            sorted_list = sorted(tmp_dict.items(), key=itemgetter(1), reverse=True)
            if len(sorted_list) > 22:
                sorted_list = sorted_list[:22]
            predict_test[i] = [x for x, y in sorted_list]
        return self.evaluate(predict_test)

    # 方法7 - 閲覧>カート>クリックで順位付け
    def method7_hybrid(self, num):
        test_ids = self.cv_tests[num]
        predict_test = {}
        for i in test_ids:
            # ユニークitem idを取得
            tmp_dict = {}
            past_items = pd.unique(self.personal_train[i]['product_id'])

            # 過去のデータから商品の重みを計算
            for j in past_items:
                tmp_dict[j] = 0
                for k in self.personal_train[i][self.personal_train[i]['product_id'] == j]['event_type']:
                    if k == 1:
                        tmp_dict[j] += 3
                    elif k == 0:
                        tmp_dict[j] += 2
                    elif k== 2:
                        tmp_dict[j] += 1
            sorted_list = sorted(tmp_dict.items(), key=itemgetter(1), reverse=True)
            if len(sorted_list) > 22:
                sorted_list = sorted_list[:22]
            predict_test[i] = [x for x, y in sorted_list]
        return self.evaluate(predict_test)

    # item-base　の　協調フィルタリング
    def method8_item_base(self,num):
        train_ids = self.cv_trains[num]
        predict_test = {}
        # item-baseの推薦は評価値行列の転置と評価値行列の内積で計算できる
        # まず評価値行列から交差検定用行列を抽出する
        slice_index=[]
        tmp_train_ids=[]
        for i in train_ids:
            if i not in self.id_dic['user_id']:
                continue
            slice_index.append(self.id_dic['user_id'].index(i))
            tmp_train_ids.append(i)

        train=self.sparse_data.tocsr()[slice_index,:]
        item_matrix=train.transpose().dot(train)

        for i in range(len(tmp_train_ids)):
            user_data=item_matrix.getrow(i).toarray()[0]
            c=zip(user_data,self.id_dic['product_id'])
            c=sorted(c,key=lambda x: x[0],reverse=True)
            sorted_list=list(zip(*c))[1]
            if len(sorted_list) > 22:
                sorted_list = sorted_list[:22]
            predict_test[tmp_train_ids[i]]=sorted_list
        return self.evaluate(predict_test)

    # Cross-validationの実行
    def CV(self):
        jobs=[]
        manager = Manager()
        score_dic = manager.dict()
        for i in range(self.K):
            p = Process(target=self.do_method, args=(i,score_dic))
            jobs.append(p)
        [x.start() for x in jobs]
        [x.join() for x in jobs]
        print('Score '+ self.name +' : {0}'.format(np.mean(list(score_dic.values()))))
        return np.mean(list(score_dic.values()))

    # 並列計算用のCV関数
    def do_method(self,data,dic):
        result = self.method_func(data)
        dic[result] = result

    #メゾッド選択用関数
    def choice_func(self,num):
        if num==None:
            print('メゾッドを選択してください')
            return -1
        if num==1:
            return self.method1_random_choice
        elif num==2:
            return self.method2_ranked_choice
        elif num==3:
            return self.method3_ranked_conversion_ignore_choice
        elif num==4:
            return self.method4_ranked_cart
        elif num==5:
            return self.method5_ranked_click
        elif num==6:
            return self.method6_ranked_view
        elif num==7:
            return self.method7_hybrid
        elif num==8:
            with open('../data/matrix/train_only_conversion_' + self.name + '.pickle', 'rb') as f:
                self.sparse_data=pickle.load(f)
            with open('../data/matrix/id_dic_only_conversion_' + self.name + '.pickle', 'rb') as f:
                self.id_dic=pickle.load(f)
            return self.method8_item_base


def work_CV(name,method):
    a=CrossValidation(name)
    a.CV(method)

def all_CV(number=5,method=None):
    print('CV開始いたします')
    scores={'A':0,'B':0,'C':0,'D':0}
    for _ in range(number):
        for i in ['A','B','C','D']:
            a=CrossValidation(i,K=5,method=method)
            scores[i]+=a.CV()
    print(str(number) + '回平均結果')
    for i in ['A', 'B', 'C', 'D']:
        scores[i]/=number
        print(i + '\t' + str(scores[i]))

    print('加重平均結果 : '+str(result_weight_mean(scores)))
    print('メゾッド選択 ：　' + str(method))

# 各カテゴリの結果を予測数の加重平均したもの
def result_weight_mean(result):
    population = {'A': 7264.0/11598, 'B': 2366.0/11598, 'C': 1648.0/11598, 'D': 320.0/11598}
    score=0
    for i in ['A','B','C','D']:
        score+=result[i]*population[i]
    return score


if __name__=='__main__':
    all_CV(1,8)