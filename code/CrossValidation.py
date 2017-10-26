#coding:utf-8
#交差検定環境の構築
#4月の30日中、24日学習して6日テストとする

import pandas as pd
import numpy as np
import scipy.sparse as sparse
from numba import jit
import datetime
import pickle
import random
import time
from multiprocessing import Process,Pool,Manager
from operator import itemgetter
from sklearn.decomposition import NMF
import tqdm
import math
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
from fastFM import als,sgd
from sklearn.preprocessing import Normalizer,StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
import csv
import GPyOpt
import cudamat as cm
from imblearn.ensemble import BalancedBaggingClassifier,BalanceCascade
import xgboost as xgb
from pprint import pprint


class CrossValidation():
    def __init__(self,name,K=5,method=None,x=None):
        self.name=name
        self.K = K
        self.read_data()
        self.split_data()
        self.method_func=self.choice_func(method)
        if method==9:
            print(x)
            n_comp, iter_num, tol = int(x[:, 0]), int(x[:, 1]), float(x[:, 2])
            model = NMF(n_components=n_comp, max_iter=iter_num, tol=tol)
            self.user_feature_matrix = model.fit_transform(self.sparse_data)
            self.item_feature_matrix = model.components_
        if method==11:
            print(x)
            self.min_weight= float(x[:,0])
            self.model = NMF(n_components=128, max_iter=1024,tol=0.001)
            self.user_feature_matrix = self.model.fit_transform(self.sparse_data)
        if method==12:
            print('learn start')
            with open('../data/fm/fm_train_X_' + self.name + '.pickle', 'rb') as f:
                r_X = pickle.load(f)
            with open('../data/fm/fm_train_y_' + self.name + '.pickle', 'rb') as f:
                r_y = pickle.load(f)

            # mask user and item
            newX=[]
            for i in r_X:
                tmp_i=i
                del(tmp_i['user'])
                del (tmp_i['item'])
                newX.append(tmp_i)

            self.v = DictVectorizer()
            X = self.v.fit_transform(newX)
            self.y = np.array(r_y).astype(np.float)
            self.raw_X = X

            # 正規化する必要がある
            #self.scaler=Normalizer()
            #self.X=self.scaler.fit_transform(X)

            self.scaler=StandardScaler(with_mean=False)
            self.scaler.fit(X)
            self.X=self.scaler.transform(X)

            print('learn finished')
        if method==13:
            with open('../data/conv_pred/train_data_' + 'A' + '.pickle', 'rb') as f:
                data = pickle.load(f)
            with open('../data/conv_pred/train_X_' + 'A' + '.pickle', 'rb') as f:
                self.name_dic_train = pickle.load(f)
            self.v = DictVectorizer()
            X = self.v.fit_transform(data['X'])
            y = np.array(data['y'])

            self.model = BalancedBaggingClassifier(n_estimators=100, n_jobs=1)
            self.model.fit(X, y)
        if method==14:
            self.nmf = NMF(n_components=128, max_iter=1024, tol=0.001)
            self.user_feature_matrix = self.nmf.fit_transform(self.sparse_data)
            with open('../data/nmf/user_feature_'+self.name+'.pickle','wb') as f:
                pickle.dump(self.user_feature_matrix,f)
            with open('../data/nmf/item_feature_'+self.name+'.pickle','wb') as f:
                pickle.dump(self.nmf.components_,f)
            if self.name!='D':
                with open('../data/conv_pred/train_data_' + self.name + '.pickle', 'rb') as f:
                    data = pickle.load(f)
                with open('../data/conv_pred/train_X_' + self.name + '.pickle', 'rb') as f:
                    self.name_dic_train = pickle.load(f)
                self.v = DictVectorizer()
                X = self.v.fit_transform(data['X'])
                y = np.array(data['y'])

                #self.model = BalancedBaggingClassifier(n_estimators=100, n_jobs=1)
                self.model = model = xgb.XGBClassifier(n_estimators=500,max_delta_step=1,scale_pos_weight=380)
                model.fit(X,y)
                #self.model.fit(X, y)
        if method==15:
            if self.name!='D':
                with open('../data/conv_pred/train_data2_' + self.name + '.pickle', 'rb') as f:
                    data = pickle.load(f)
                with open('../data/conv_pred/train_X2_' + self.name + '.pickle', 'rb') as f:
                    self.name_dic_train = pickle.load(f)
                self.v = DictVectorizer()
                self.select=[0,1,2,3,4,7,8,9,14,15,16,17,18,19,21,22,23,24,25,26]

                X = self.v.fit_transform(data['X'])[:,self.select]
                pprint(self.v.feature_names_)
                print(len(self.v.feature_names_))
                y = np.array(data['y'])
                self.forest = BalancedBaggingClassifier(n_estimators=100, n_jobs=1)
                self.forest.fit(X, y)
        if method==16:
            with open('../data/nmf/user_feature_'+self.name+'.pickle','rb') as f:
                self.user_feature_matrix=pickle.load(f)
            with open('../data/nmf/item_feature_'+self.name+'.pickle','rb') as f:
                self.item_feature_matrix=pickle.load(f)
            with open('../data/conv_pred/train_data_notRec_' + self.name + '.pickle', 'rb') as f:
                data = pickle.load(f)
            with open('../data/conv_pred/train_X_' + self.name + '.pickle', 'rb') as f:
                self.name_dic_train = pickle.load(f)
            self.v = DictVectorizer()
            X = self.v.fit_transform(data['X'])
            y = np.array(data['y'])
            self.forest = BalancedBaggingClassifier(n_estimators=100, n_jobs=1)
            self.forest.fit(X, y)

    #データを読み込み分割
    def read_data(self):
        #個人のデータ読み込み
        self.personal_result=pd.read_pickle('../data/personal/personal_test_items_IDCG_'+self.name+'.pickle')
        self.personal_train=pd.read_pickle('../data/personal/personal_train_' + self.name + '.pickle')
        with open('../data/matrix/train_optimized_' + self.name + '.pickle', 'rb') as f:
            self.sparse_data = pickle.load(f)
        with open('../data/matrix/id_dic_optimized_' + self.name + '.pickle', 'rb') as f:
            self.id_dic = pickle.load(f)
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
        # 重複を除く
        new_items=[]
        for i in items:
            if i not in new_items:
                new_items.append(i)
        output=[]
        for i in range(len(new_items)):
            if new_items[i] in list(self.personal_result[user_id].keys()):
                DCG+=(2**self.personal_result[user_id][new_items[i]]-1)/np.log2(i+2)
        #         output.append(self.personal_result[user_id][new_items[i]])
        #     else:
        #         output.append(0)
        # print(output)
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

    #含有率を調査する関数
    def analysis_content(self, predict,important):
        output=[]
        for user_id in predict.keys():
            for item in predict[user_id]:
                # もしtest期間に含まれていれば
                if item in list(self.personal_result[user_id].keys()):
                    output.append(important)
        with open('../data/view/analysis_content_'+self.name+'.pickle','wb') as f:
            pickle.dump(output,f)

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

    # 方法8 - item-base　の　協調フィルタリング
    def method8_item_base(self,num):
        test_ids = self.cv_tests[num]
        # item-baseの推薦は評価値行列の転置と評価値行列の内積で計算できる
        train=self.sparse_data.tocsr()
        item_matrix=train.transpose().dot(train)
        # なぜか上の演算でcscになっているのでcsrに治す
        item_matrix=item_matrix.tocsr()
        sorted_list = None

        # 方法7を用いた商品推薦
        predict_test = {}
        for i in tqdm.tqdm(test_ids):
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
                    elif k == 2:
                        tmp_dict[j] += 1
            sorted_list = sorted(tmp_dict.items(), key=itemgetter(1), reverse=True)
            if len(sorted_list) > 22:
                sorted_list = sorted_list[:22]
            predict_test[i] = [x for x, y in sorted_list]

            # 過去の推薦個数が22に満たなければ
            if len(predict_test[i])<22 and len(predict_test[i])>0:
                c_fil_num=int(1+math.floor((22-len(predict_test[i]))/len(predict_test[i])))
                # 過去からの推薦の各商品について
                c_fil_items=[]
                for j in predict_test[i]:
                    # itemの推薦順序を確保
                    t=self.id_dic['product_id'].index(j)
                    #item_base=item_matrix.getrow(t)
                    item_base = item_matrix[t]
                    item_base=item_base.toarray()[0]
                    # 上位23件を持ってくる
                    ind = np.argpartition(item_base, int(-23*c_fil_num))[int(-23*c_fil_num):]
                    rec_item = []
                    # 降順に並べ替えて（今まで昇順でやってた・・・・・・・・・）
                    for item in ind[np.argsort(item_base[ind])[::-1]]:
                        item=self.id_dic['product_id'][item]
                        if item not in predict_test and item not in c_fil_items:
                            rec_item.append(item)
                        if len(rec_item)==c_fil_num:
                            break
                    #assert len(rec_item)!=c_fil_num, 'No Recommend items'
                    c_fil_items.extend(rec_item)
                predict_test[i].extend(c_fil_items)
                if len(predict_test[i])>22:
                    predict_test[i]=predict_test[i][:22]
        return self.evaluate(predict_test)

    # 方法9 - NMFのみを用いた推薦
    def method9_NMF_only(self, num):
        # GPUを用いて行列演算（初期化）
        cm.cuda_set_device(0)
        cm.init()
        test_ids = self.cv_tests[num]
        predict_test = {}
        # item-baseの推薦は評価値行列の転置と評価値行列の内積で計算できる


        for i in tqdm.tqdm(test_ids):
            if i not in self.id_dic['user_id']:
                continue
            #user_unique_train=pd.unique(self.personal_train[i]['product_id'])
            #est_user_eval=np.dot(user_feature_matrix[self.id_dic['user_id'].index(i)],item_feature_matrix)
            index_user = self.id_dic['user_id'].index(i)
            nmf_result = cm.dot(cm.CUDAMatrix(self.user_feature_matrix[index_user:index_user+1]), cm.CUDAMatrix(self.item_feature_matrix)).asarray()
            est_user_eval = nmf_result[0]
            tmp=sorted(zip(est_user_eval,self.id_dic['product_id']),key=lambda x:x[0],reverse=True)
            predict=list(zip(*tmp))[1]
            out_dic=[]
            for j in predict:
                #if j not in user_unique_train:
                out_dic.append(j)
                if len(out_dic)==22:
                    break
            predict_test[i]=out_dic
        return self.evaluate(predict_test)

    # 方法10 - 方法7に時間重みを加えた推薦法
    def method10_time_weight(self, num):
        # ベイズ最適化結果の読み込み
        parm_dic={'A': {'conv': -100, 'click': 0, 'view': 0.95845924, 'cart': 0.22327048},
             'B': {'conv': 1, 'click': 0.43314098, 'view': 0.5480186, 'cart': 1},
             'C': {'conv': 0, 'click': 0, 'view': 0.71978554, 'cart': 1},
             'D': {'conv': 1, 'click': 0, 'view': 0.82985685, 'cart': 0}}

        with open('../data/time_weight/fitting_balanced_' + self.name + '.pickle', 'rb') as f:
            time_weight=pickle.load(f)
        test_min = datetime.datetime(year=2017, month=4, day=24)
        test_ids = self.cv_tests[num]
        predict_test = {}
        importance={}
        for i in tqdm.tqdm(test_ids):
            # ユニークitem idを取得
            tmp_dict = {}
            past_items = pd.unique(self.personal_train[i]['product_id'])

            # 過去のデータから商品の重みを計算
            for j in past_items:
                tmp_dict[j] = 0
                for _,row in self.personal_train[i][self.personal_train[i]['product_id'] == j].iterrows():
                    if row['event_type'] == 3:
                        tmp_dict[j] += parm_dic[self.name]['conv'] * time_weight[-1*(row['time_stamp']-test_min).days]
                    elif row['event_type'] == 2:
                        tmp_dict[j] += parm_dic[self.name]['click'] * time_weight[-1*(row['time_stamp']-test_min).days]
                    elif row['event_type'] == 1:
                        tmp_dict[j] += parm_dic[self.name]['view'] * time_weight[-1*(row['time_stamp']-test_min).days]
                    elif row['event_type'] == 0:
                        tmp_dict[j] += parm_dic[self.name]['cart'] * time_weight[-1*(row['time_stamp']-test_min).days]

            sorted_list = sorted(tmp_dict.items(), key=itemgetter(1), reverse=True)
            if len(sorted_list) > 22:
                sorted_list = sorted_list[:22]
            predict_test[i] = [x for x, y in sorted_list]
            importance[i] = [y for x, y in sorted_list]
        return self.evaluate(predict_test)

    # 方法11 - 過去と協調のハイブリッド推薦手法
    def method11_past_and_collaborate(self, num):
        # GPUを用いて行列演算（初期化）
        cm.cuda_set_device(0)
        cm.init()
        # NMFで推薦する個数
        nmf_number_min=0
        item_feature_matrix = self.model.components_

        with open('../data/view/analysis_content_' + self.name + '.pickle', 'rb') as f:
            time_r_dic = pickle.load(f)
        test_ids = self.cv_tests[num]
        predict_test = {}
        for i in tqdm.tqdm(test_ids):
            if i not in self.id_dic['user_id']:
                continue
            nmf_number = nmf_number_min
            # ユニークitem idを取得
            tmp_dict = {}
            if i not in time_r_dic.keys():
                continue

            past_r_id = time_r_dic[i]['items']
            past_r_im = time_r_dic[i]['importance']

            # if len(sorted_list)>=self.min_num:
            #     sorted_list = []
            #     for i_rank in range(len(time_r_dic[i]['items'])):
            #         if time_r_dic[i]['importance'][i_rank]>self.min_weight:
            #             sorted_list.append(time_r_dic[i]['items'][i_rank])

            #est_user_eval = np.dot(self.user_feature_matrix[self.id_dic['user_id'].index(i)], item_feature_matrix)
            index_user=self.id_dic['user_id'].index(i)
            nmf_result = cm.dot(cm.CUDAMatrix(self.user_feature_matrix[index_user:index_user + 1]),cm.CUDAMatrix(item_feature_matrix)).asarray()
            est_user_eval=nmf_result[0]

            # 2-way marge
            est_user_eval*=self.min_weight
            for tmp_id in past_r_id:
                est_user_eval[self.id_dic['product_id'].index(tmp_id)]+=past_r_im[past_r_id.index(tmp_id)]


            tmp = sorted(zip(est_user_eval, self.id_dic['product_id']), key=lambda x: x[0], reverse=True)
            predict = list(zip(*tmp))[1]

            # add_list=[]
            # num=0
            # while len(add_list)!=nmf_number:
            #     if predict[num] not in sorted_list:
            #         add_list.append(predict[num])
            #     num+=1
            # if len(sorted_list) > 22 - nmf_number:
            #     sorted_list = sorted_list[:22-nmf_number]
            # sorted_list.extend(add_list)
            predict_test[i] = predict[:22]
        return self.evaluate(predict_test)

    # 方法12 - FMを用いた推薦法
    def method12_fm_original(self,num,x):
        print(x)
        w, p_v, l2, iter, rank = float(x[:, 0]), float(x[:, 1]), float(x[:, 2]), int(x[:, 3]), int(x[:, 4])
        self.fm = als.FMRegression(n_iter=iter, rank=rank, l2_reg_w=w, l2_reg_V=p_v,l2_reg=l2)
        self.fm.fit(self.X, self.y)
        with open('../data/time_weight/fitting_balanced_' + self.name + '.pickle', 'rb') as f:
            time_weight=pickle.load(f)
        test_ids = self.cv_tests[num]
        test_ids=test_ids[:100]
        test_min = datetime.datetime(year=2017, month=4, day=24)
        predict_test = {}
        for user in tqdm.tqdm(test_ids):
            # ユニークitem idを取得
            tmp_dict = {}

            past_items = pd.unique(self.personal_train[user]['product_id'])
            for item in past_items:
                #user_dic = {'user': user,'item':item,'cart':0,'view':0,'click':0,'conv':0,'first_day':0,'last_day':30,'event_num':0}
                user_dic = {'cart': 0, 'view': 0, 'click': 0, 'conv': 0, 'first_day': 0,
                            'last_day': 30, 'event_num': 0}
                item_list=[]
                input_list=[]
                for _, row in self.personal_train[user][self.personal_train[user]['product_id'] == item].iterrows():
                    user_dic['event_num']+=1

                    day=-1 * (row['time_stamp'] - test_min).days

                    if row['event_type'] == 0:
                        user_dic['cart'] += 1 * time_weight[day]
                    elif row['event_type'] == 1:
                        user_dic['view'] += 1 * time_weight[day]
                    elif row['event_type'] == 2:
                        user_dic['click'] += 1 * time_weight[day]
                    elif row['event_type'] == 3:
                        user_dic['conv'] += 1 * time_weight[day]

                    if day > user_dic['first_day']:
                        user_dic['first_day']=day
                    if day < user_dic['last_day']:
                        user_dic['last_day']=day

                # standalization
                a=self.v.transform([user_dic])
                a=self.scaler.transform(a)
                p=self.fm.predict(a)
                tmp_dict[item]=p[0]
            sorted_list = sorted(tmp_dict.items(), key=itemgetter(1), reverse=True)
            if len(sorted_list) > 22:
                sorted_list = sorted_list[:22]
            predict_test[user] = [x for x, y in sorted_list]
        return self.evaluate(predict_test)

    # 方法13 - MLを用いた推薦法
    def method13_random_forest(self, num):
        # ベイズ最適化結果の読み込み
        parm_dic = {'A': {'conv': -100, 'click': 0, 'view': 0.95845924, 'cart': 0.22327048},
                    'B': {'conv': 1, 'click': 0.43314098, 'view': 0.5480186, 'cart': 1},
                    'C': {'conv': 0, 'click': 0, 'view': 0.71978554, 'cart': 1},
                    'D': {'conv': 1, 'click': 0, 'view': 0.82985685, 'cart': 0}}

        with open('../data/time_weight/fitting_balanced_' + self.name + '.pickle', 'rb') as f:
            time_weight = pickle.load(f)
        test_min = datetime.datetime(year=2017, month=4, day=24)
        test_ids = self.cv_tests[num]
        predict_test = {}
        importance = {}
        for i in tqdm.tqdm(test_ids):
            # ユニークitem idを取得
            tmp_dict = {}
            past_items = pd.unique(self.personal_train[i]['product_id'])

            # 過去のデータから商品の重みを計算
            for j in past_items:
                tmp_dict[j] = 0
                for _, row in self.personal_train[i][self.personal_train[i]['product_id'] == j].iterrows():
                    if row['event_type'] == 3:
                        tmp_dict[j] += parm_dic[self.name]['conv'] * time_weight[
                            -1 * (row['time_stamp'] - test_min).days]
                    elif row['event_type'] == 2:
                        tmp_dict[j] += parm_dic[self.name]['click'] * time_weight[
                            -1 * (row['time_stamp'] - test_min).days]
                    elif row['event_type'] == 1:
                        tmp_dict[j] += parm_dic[self.name]['view'] * time_weight[
                            -1 * (row['time_stamp'] - test_min).days]
                    elif row['event_type'] == 0:
                        tmp_dict[j] += parm_dic[self.name]['cart'] * time_weight[
                            -1 * (row['time_stamp'] - test_min).days]

            sorted_list = sorted(tmp_dict.items(), key=itemgetter(1), reverse=True)
            sorted_list = [x for x, y in sorted_list]
            sorted_list2=[]
            #importance[i] = [y for x, y in sorted_list]

            # randamforest input data
            input_data=[]
            for k in sorted_list:
                if len(self.name_dic_train[i][k]) != 0:
                    sorted_list2.append(k)
                    input_data.append(self.name_dic_train[i][k])
            if len(input_data)!=0:
                X=self.v.transform(input_data)
                pred=self.model.predict_proba(X)
                pred=pred[:,1]
                conv_list=[]
                mysort=sorted(zip(sorted_list2,pred),key=lambda x:x[1],reverse=True)
                for k in range(len(mysort)):
                    if mysort[k][1]>=0.8:
                        conv_list.append(mysort[k][0])
                for k in sorted_list:
                    if k not in conv_list:
                        conv_list.append(k)
                sorted_list=conv_list

            if len(sorted_list) > 22:
                sorted_list = sorted_list[:22]
            predict_test[i]=sorted_list

        return self.evaluate(predict_test)

    # 方法14 - 過去と協調のハイブリッド+RandomForest推薦手法
    def method14_past_and_collaborate(self, num):
        # GPUを用いて行列演算（初期化）
        # cm.cuda_set_device(0)
        # cm.init()
        parm_dic = {'A': {'conv': 0, 'click': 0, 'view': 0.95845924, 'cart': 0.22327048},
                    'B': {'conv': 1, 'click': 0.43314098, 'view': 0.5480186, 'cart': 1},
                    'C': {'conv': 0, 'click': 0, 'view': 0.71978554, 'cart': 1},
                    'D': {'conv': 1, 'click': 0, 'view': 0.82985685, 'cart': 0}}

        with open('../data/time_weight/fitting_balanced_' + self.name + '.pickle', 'rb') as f:
            time_weight = pickle.load(f)
        test_min = datetime.datetime(year=2017, month=4, day=24)
        item_feature_matrix = self.nmf.components_
        test_ids = self.cv_tests[num]
        predict_test = {}
        for i in tqdm.tqdm(test_ids):
            if i not in self.id_dic['user_id']:
                continue

            # ユニークitem idを取得
            tmp_dict = {}
            past_items = pd.unique(self.personal_train[i]['product_id'])

            # 過去のデータから商品の重みを計算
            for j in past_items:
                tmp_dict[j] = self.name_dic_train[i][j]['score_conv']*parm_dic[self.name]['conv']+self.name_dic_train[i][j]['score_click']*parm_dic[self.name]['click']+self.name_dic_train[i][j]['score_view']*parm_dic[self.name]['view']+self.name_dic_train[i][j]['score_cart']*parm_dic[self.name]['cart']

            sorted_list = sorted(tmp_dict.items(), key=itemgetter(1), reverse=True)
            sorted_list = [x for x, y in sorted_list]
            sorted_list2 = []
            input_data = []
            for k in sorted_list:
                if len(self.name_dic_train[i][k]) != 0:
                    sorted_list2.append(k)
                    input_data.append(self.name_dic_train[i][k])
            if len(input_data) != 0 and self.name!='D':
                X = self.v.transform(input_data)
                pred = self.model.predict_proba(X)
                pred = pred[:, 1]
                conv_list = []
                mysort = sorted(zip(sorted_list2, pred), key=lambda x: x[1], reverse=True)
                for k in range(len(mysort)):
                    if mysort[k][1] > 0.5:
                        conv_list.append(mysort[k][0])
                for k in sorted_list:
                    if k not in conv_list:
                        conv_list.append(k)
                sorted_list = conv_list
            if len(sorted_list) > 22:
                sorted_list = sorted_list[:22]

            amari=22-len(sorted_list)
            if amari>0:
                # index_user = self.id_dic['user_id'].index(i)
                # nmf_result = cm.dot(cm.CUDAMatrix(self.user_feature_matrix[index_user:index_user + 1]),
                #                     cm.CUDAMatrix(item_feature_matrix)).asarray()
                # est_user_eval = nmf_result[0]
                est_user_eval = np.dot(self.user_feature_matrix[self.id_dic['user_id'].index(i)], item_feature_matrix)

                # 2-way marge
                nmf_sort=list(zip(*sorted(zip(self.id_dic['product_id'],est_user_eval),key=lambda x:x[1],reverse=True)))[0]
                sorted_list.extend(nmf_sort[:amari])

            predict_test[i] = sorted_list
        return self.evaluate(predict_test)

    # 方法15 - Predict hikaku
    def method15_predict_analyze(self, num):
        parm_dic = {'A': {'conv': 0, 'click': 0, 'view': 0.95845924, 'cart': 0.22327048},
                    'B': {'conv': 1, 'click': 0.43314098, 'view': 0.5480186, 'cart': 1},
                    'C': {'conv': 0, 'click': 0, 'view': 0.71978554, 'cart': 1},
                    'D': {'conv': 1, 'click': 0, 'view': 0.82985685, 'cart': 0}}
        test_ids = self.cv_tests[num]
        predict_test = {}
        for i in tqdm.tqdm(test_ids):
            if i not in self.id_dic['user_id']:
                continue

            # ユニークitem idを取得
            tmp_dict = {}
            past_items = pd.unique(self.personal_train[i]['product_id'])

            # 過去のデータから商品の重みを計算
            for j in past_items:
                tmp_dict[j] = self.name_dic_train[i][j]['score_conv'] * parm_dic[self.name]['conv'] + \
                              self.name_dic_train[i][j]['score_click'] * parm_dic[self.name]['click'] + \
                              self.name_dic_train[i][j]['score_view'] * parm_dic[self.name]['view'] + \
                              self.name_dic_train[i][j]['score_cart'] * parm_dic[self.name]['cart']

            sorted_list = sorted(tmp_dict.items(), key=itemgetter(1), reverse=True)
            sorted_list = [x for x, y in sorted_list]
            sorted_list2 = []
            input_data = []
            for k in sorted_list:
                if len(self.name_dic_train[i][k]) != 0:
                    sorted_list2.append(k)
                    input_data.append(self.name_dic_train[i][k])
            if len(input_data) != 0 and self.name != 'D':
                X = self.v.transform(input_data)[:,self.select]
                #pred = self.model.predict_proba(X)
                pred = self.forest.predict_proba(X)[:, 1]
                conv_list = []
                mysort = sorted(zip(sorted_list2, pred), key=lambda x: x[1], reverse=True)
                #print(mysort)
                for k in range(len(mysort)):
                    if mysort[k][1] > 0:
                        conv_list.append(mysort[k][0])
                for k in sorted_list:
                    if k not in conv_list:
                        conv_list.append(k)
                sorted_list = conv_list
            if len(sorted_list) > 22:
                sorted_list = sorted_list[:22]

            predict_test[i] = sorted_list
        return self.evaluate(predict_test)

    # 方法15 - Predict hikaku
    def method16_not_recommend(self, num):
        parm_dic = {'A': {'conv': 0, 'click': 0, 'view': 0.95845924, 'cart': 0.22327048},
                    'B': {'conv': 1, 'click': 0.43314098, 'view': 0.5480186, 'cart': 1},
                    'C': {'conv': 0, 'click': 0, 'view': 0.71978554, 'cart': 1},
                    'D': {'conv': 1, 'click': 0, 'view': 0.82985685, 'cart': 0}}
        test_ids = self.cv_tests[num]
        predict_test = {}
        for i in tqdm.tqdm(test_ids):
            if i not in self.id_dic['user_id']:
                continue

            # ユニークitem idを取得
            tmp_dict = {}
            past_items = pd.unique(self.personal_train[i]['product_id'])

            # 過去のデータから商品の重みを計算
            for j in past_items:
                tmp_dict[j] = self.name_dic_train[i][j]['score_conv'] * parm_dic[self.name]['conv'] + \
                              self.name_dic_train[i][j]['score_click'] * parm_dic[self.name]['click'] + \
                              self.name_dic_train[i][j]['score_view'] * parm_dic[self.name]['view'] + \
                              self.name_dic_train[i][j]['score_cart'] * parm_dic[self.name]['cart']

            sorted_list = sorted(tmp_dict.items(), key=itemgetter(1), reverse=True)
            sorted_list = [x for x, y in sorted_list]
            sorted_list2 = []
            rec_list = []
            input_data = []
            for k in sorted_list:
                if len(self.name_dic_train[i][k]) != 0:
                    sorted_list2.append(k)
                    input_data.append(self.name_dic_train[i][k])
            if len(input_data) != 0 and self.name != 'D':
                X = self.v.transform(input_data)
                pred = self.forest.predict_proba(X)[:, 1]
                conv_list = []
                mysort = sorted(zip(sorted_list2, pred), key=lambda x: x[1], reverse=True)
                # print(mysort)
                for k in range(len(mysort)):
                    if mysort[k][1] > 5.5:
                        conv_list.append(mysort[k][0])
                for k in sorted_list:
                    if k not in conv_list:
                        rec_list.append(k)
                sorted_list = rec_list
            if len(sorted_list) >= 22:
                sorted_list = sorted_list[:22]
            else:
                est_user_eval = np.dot(self.user_feature_matrix[self.id_dic['user_id'].index(i)], self.item_feature_matrix)
                tmp = sorted(zip(est_user_eval, self.id_dic['product_id']), key=lambda x: x[0], reverse=True)

                c_num=0
                while len(sorted_list)<22:
                    if tmp[c_num][1] not in sorted_list:
                        sorted_list.append(tmp[c_num][1])
                    c_num+=1

            predict_test[i] = sorted_list
        return self.evaluate(predict_test)

    # Cross-validationの実行
    def CV_multi(self):
        jobs=[]
        manager = Manager()
        score_dic = manager.dict()
        for i in range(self.K):
            p = Process(target=self.do_method, args=(i,score_dic))
            jobs.append(p)
        [x.start() for x in jobs]
        [x.join() for x in jobs]
        print('Score '+ self.name +' : {0}'.format(np.mean(list(score_dic.values()))))
        return -1*np.mean(list(score_dic.values()))

    # Cross-validationの実行
    def CV_normal(self):
        for i in range(self.K):
            a=self.method_func(i)
        print('Score ' + self.name + ' : {0}'.format(a))
        return a

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
            return self.method8_item_base
        elif num==9:
            return self.method9_NMF_only
        elif num == 10:
            return self.method10_time_weight
        elif num == 11:
            return self.method11_past_and_collaborate
        elif num == 12:
            return self.method12_fm_original
        elif num == 13:
            return self.method13_random_forest
        elif num==14:
            return self.method14_past_and_collaborate
        elif num==15:
            return self.method15_predict_analyze
        elif num==16:
            return self.method16_not_recommend
        else:
            print('メゾッドを選択してください')
            return -1

def all_CV(number=5,method=None):
    #names=['A', 'B', 'C', 'D']
    names=['A','B','C']
    print('CV開始いたします')
    scores={'A':0,'B':0,'C':0,'D':0}
    for _ in range(number):
        for i in names:
            a=CrossValidation(i,K=5,method=method)
            scores[i]+=a.CV_multi()
    print(str(number) + '回平均結果')
    for i in names:
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

def baysian_optimazation_for_fm():
    bounds = [{'name': 'min_weight', 'type': 'continuous', 'domain': (0,10)}]



    # 事前探索を行います。
    opt_mnist = GPyOpt.methods.BayesianOptimization(f=bay_opt, domain=bounds,initial_design_numdata=5,verbosity=True)

    # 最適なパラメータを探索します。
    opt_mnist.run_optimization(max_iter=20,verbosity=True)
    print("optimized parameters: {0}".format(opt_mnist.x_opt))
    print("optimized loss: {0}".format(opt_mnist.fx_opt))
def bay_opt(x):
    a = CrossValidation('A', K=5, method=15,x=x)
    return a.CV_multi()


if __name__=='__main__':
    all_CV(1,16)
    #baysian_optimazation_for_fm()