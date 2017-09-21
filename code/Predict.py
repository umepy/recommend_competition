#coding:utf-8
#予測用のクラス

import pandas as pd
import pickle
import random
import numpy as np
import time
from numba import jit
from operator import itemgetter
import tqdm
import math

class Predict():
    def __init__(self):
        self.read_data()
    def read_data(self):
        #個人のデータ読み込み
        self.personal_train={}
        self.sparse_data={}
        self.id_dic={}
        for name in ['A','B','C','D']:
            with open('../data/personal/personal_'+name+'.pickle','rb') as f:
                self.personal_train[name]=pickle.load(f)
            with open('../data/matrix/all_weighted_' + name + '.pickle', 'rb') as f:
                self.sparse_data[name] = pickle.load(f)
            with open('../data/matrix/all_id_dic_weighted_' + name + '.pickle', 'rb') as f:
                self.id_dic[name] = pickle.load(f)

        #予測用idの読み込み
        with open('../data/submit_ids.pickle','rb') as f:
            self.submit_ids=pickle.load(f)
    def method1_choice_from_past_data(self,name,test_ids):
        predict_test={}
        for i in test_ids:
            #ユニークitem idを取得
            past_items=pd.unique(self.personal_train[name][i]['product_id'])
            random.shuffle(past_items)
            if len(past_items) > 22:
                past_items=past_items[:22]
            predict_test[i]=past_items
        return predict_test

    def method6_ranked_view(self, name, test_ids):
        predict_test = {}
        for i in tqdm.tqdm(test_ids):
            # ユニークitem idを取得
            tmp_dict = {}
            past_items = pd.unique(self.personal_train[name][i]['product_id'])

            # 過去のデータから商品の重みを計算
            for j in past_items:
                tmp_dict[j] = 0
                for k in self.personal_train[name][i][self.personal_train[name][i]['product_id'] == j]['event_type']:
                    if k == 1:
                        tmp_dict[j] += 1
                    if k == -1:
                        del tmp_dict[j]
                        break
            sorted_list = sorted(tmp_dict.items(), key=itemgetter(1), reverse=True)
            if len(sorted_list) > 22:
                sorted_list = sorted_list[:22]
            predict_test[i] = [x for x, y in sorted_list]
        return predict_test

    def method7_hybrid(self, name, test_ids):
        predict_test = {}
        for i in tqdm.tqdm(test_ids):
            # ユニークitem idを取得
            tmp_dict = {}
            past_items = pd.unique(self.personal_train[name][i]['product_id'])

            # 過去のデータから商品の重みを計算
            for j in past_items:
                tmp_dict[j] = 0
                for k in self.personal_train[name][i][self.personal_train[name][i]['product_id'] == j]['event_type']:
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
        return predict_test

    def method8_itembase(self, name, test_ids):
        go_num=0
        predict_test = {}
        train = self.sparse_data[name].tocsr()
        item_matrix = train.transpose().dot(train)
        # なぜか上の演算でcscになっているのでcsrに治す
        item_matrix = item_matrix.tocsr()

        for i in test_ids:
            # ユニークitem idを取得
            tmp_dict = {}
            past_items = pd.unique(self.personal_train[name][i]['product_id'])

            # 過去のデータから商品の重みを計算
            for j in past_items:
                tmp_dict[j] = 0
                for k in self.personal_train[name][i][self.personal_train[name][i]['product_id'] == j]['event_type']:
                    if k == 1:
                        tmp_dict[j] += 3
                    elif k == 0:
                        tmp_dict[j] += 2
                    elif k == 2:
                        tmp_dict[j] += 1
            sorted_list = sorted(tmp_dict.items(), key=itemgetter(1), reverse=True)
            if len(sorted_list) > 22:
                sorted_list = sorted_list[:22]
            predict_test[i] = [x for x, y in sorted_list]

            # 過去の推薦個数が22に満たなければ
            if len(predict_test[i]) < 22 and len(predict_test[i]) > 0:
                go_num+=1
                c_fil_num = int(1 + math.floor((22 - len(predict_test[i])) / len(predict_test[i])))
                # 過去からの推薦の各商品について
                c_fil_items = []
                for j in predict_test[i]:
                    # itemの推薦順序を確保
                    t = self.id_dic[name]['product_id'].index(j)
                    # item_base=item_matrix.getrow(t)
                    item_base = item_matrix[t]
                    item_base = item_base.toarray()[0]
                    # 上位100件を持ってくる
                    ind = np.argpartition(item_base, -100)[-100:]
                    rec_item = []
                    # 降順に並べ替えて（今まで昇順でやってた・・・・・・・・・）
                    for item in ind[np.argsort(item_base[ind])[::-1]]:
                        item = self.id_dic[name]['product_id'][item]
                        if item not in predict_test and item not in c_fil_items:
                            rec_item.append(item)
                        if len(rec_item) == c_fil_num:
                            break
                    # assert len(rec_item)!=c_fil_num, 'No Recommend items'
                    c_fil_items.extend(rec_item)
                predict_test[i].extend(c_fil_items)
                if len(predict_test[i]) > 22:
                    predict_test[i] = predict_test[i][:22]
            print(go_num)
        return predict_test

    def all_predict(self):
        print('予測開始します')
        predict_ids={}
        for i in ['A','B','C','D']:
            print(i)
            predict_ids[i]=self.method8_itembase(i, self.submit_ids[i])

        submit_list=[]
        for i in ['A','B','C','D']:
            for j in self.submit_ids[i]:
                for k in range(len(predict_ids[i][j])):
                    tmp_list = [j]
                    tmp_list.append(predict_ids[i][j][k])
                    tmp_list.append(k)
                    submit_list.append(tmp_list)
        df=pd.DataFrame(submit_list)
        df.to_csv('../data/submit/submit.tsv',sep='\t',header=False,index=False)

if __name__=='__main__':
    a=Predict()
    a.all_predict()