#coding:utf-8
#予測用のクラス

import pandas as pd
import pickle
import random
import time
from numba import jit
from operator import itemgetter
import tqdm

class Predict():
    def __init__(self):
        self.read_data()
    def read_data(self):
        #個人のデータ読み込み
        self.personal_train={}
        for name in ['A','B','C','D']:
            with open('../data/personal/personal_'+name+'.pickle','rb') as f:
                self.personal_train[name]=pickle.load(f)

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

    def all_predict(self):
        print('予測開始します')
        predict_ids={}
        for i in ['A','B','C','D']:
            print(i)
            predict_ids[i]=self.method6_ranked_view(i, self.submit_ids[i])

        submit_list=[]
        tmp_list=[]
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