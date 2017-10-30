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
import datetime
from sklearn.decomposition import NMF
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.feature_extraction import DictVectorizer
import cudamat as cm

class Predict():
    def __init__(self):
        self.read_data()
        random.seed(1234)
    def read_data(self):
        #個人のデータ読み込み
        self.personal_train={}
        self.sparse_data={}
        self.id_dic={}
        for name in ['A','B','C','D']:
            self.personal_train[name]=pd.read_pickle('../data/personal/personal_'+name+'.pickle')
            if name=='D':
                continue
            with open('../data/matrix/all_optimized_' + name + '.pickle', 'rb') as f:
                self.sparse_data[name] = pickle.load(f)
            with open('../data/matrix/all_id_dic_optimized_' + name + '.pickle', 'rb') as f:
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

    # 方法10 - 方法7に時間重みを加えた推薦法
    def method10_time_weight(self, name, test_ids):
        parm_dic = {'A': {'conv': 0, 'click': 0.20701892, 'view': 0.78720054, 'cart': 0.19557122},
                    'B': {'conv': 1, 'click': 0.43314098, 'view': 0.5480186, 'cart': 1},
                    'C': {'conv': 0, 'click': 0, 'view': 0.71978554, 'cart': 1},
                    'D': {'conv': 1, 'click': 0, 'view': 0.82985685, 'cart': 0}}
        with open('../data/time_weight/fitting_balanced_' + name + '.pickle', 'rb') as f:
            time_weight = pickle.load(f)
        test_min = datetime.datetime(year=2017, month=5, day=1)
        predict_test = {}
        for i in tqdm.tqdm(test_ids):
            # ユニークitem idを取得
            tmp_dict = {}
            past_items = pd.unique(self.personal_train[name][i]['product_id'])

            # 過去のデータから商品の重みを計算
            for j in past_items:
                tmp_dict[j] = 0
                for _, row in self.personal_train[name][i][self.personal_train[name][i]['product_id'] == j].iterrows():
                    if row['event_type'] == 0:
                        tmp_dict[j] += parm_dic[name]['cart'] * time_weight[-1 * (row['time_stamp'] - test_min).days]
                    elif row['event_type'] == 1:
                        tmp_dict[j] += parm_dic[name]['view'] * time_weight[-1 * (row['time_stamp'] - test_min).days]
                    elif row['event_type'] == 2:
                        tmp_dict[j] += parm_dic[name]['click'] * time_weight[-1 * (row['time_stamp'] - test_min).days]
                    elif row['event_type'] == 3:
                        tmp_dict[j] += parm_dic[name]['conv'] * time_weight[-1 * (row['time_stamp'] - test_min).days]

            sorted_list = sorted(tmp_dict.items(), key=itemgetter(1), reverse=True)
            if len(sorted_list) > 22:
                sorted_list = sorted_list[:22]
            predict_test[i] = [x for x, y in sorted_list]
        return predict_test

    # 方法11 - 過去とNMFのハイブリッド推薦
    def method11_past_and_collaborate(self, name, test_ids):
        with open('../data/time_weight/fitting_balanced_' + name + '.pickle', 'rb') as f:
            time_weight = pickle.load(f)
        parm_dic = {'A': {'conv': 0, 'click': 0.20701892, 'view': 0.78720054, 'cart': 0.19557122},
                    'B': {'conv': 1, 'click': 0.43314098, 'view': 0.5480186, 'cart': 1},
                    'C': {'conv': 0, 'click': 0, 'view': 0.71978554, 'cart': 1},
                    'D': {'conv': 1, 'click': 0, 'view': 0.82985685, 'cart': 0}}
        if name != 'D':
            with open('../data/matrix/all_time_weighted_' + name + '.pickle', 'rb') as f:
                sparse_data = pickle.load(f)
            with open('../data/matrix/all_id_dic_time_weighted_' + name + '.pickle', 'rb') as f:
                id_dic = pickle.load(f)
            model = NMF(n_components=128,max_iter=1024,tol=0.001)
            user_feature_matrix = model.fit_transform(sparse_data)
            item_feature_matrix = model.components_
            nmf_number_min=0
        test_min = datetime.datetime(year=2017, month=5, day=1)
        predict_test = {}
        for i in tqdm.tqdm(test_ids):
            # ユニークitem idを取得
            tmp_dict = {}
            past_items = pd.unique(self.personal_train[name][i]['product_id'])

            # 過去のデータから商品の重みを計算
            for j in past_items:
                tmp_dict[j] = 0
                for _, row in self.personal_train[name][i][
                            self.personal_train[name][i]['product_id'] == j].iterrows():
                    if row['event_type'] == 1:
                        tmp_dict[j] += parm_dic[name]['view'] * time_weight[-1 * (row['time_stamp'] - test_min).days]
                    elif row['event_type'] == 0:
                        tmp_dict[j] += parm_dic[name]['cart'] * time_weight[-1 * (row['time_stamp'] - test_min).days]
                    elif row['event_type'] == 2:
                        tmp_dict[j] += parm_dic[name]['click'] * time_weight[-1 * (row['time_stamp'] - test_min).days]
                    elif row['event_type'] == 3:
                        tmp_dict[j] += parm_dic[name]['conv'] * time_weight[-1 * (row['time_stamp'] - test_min).days]

            sorted_list = sorted(tmp_dict.items(), key=itemgetter(1), reverse=True)
            sorted_list = [x for x, y in sorted_list]
            if name=='D':
                if len(sorted_list) > 22:
                    sorted_list = sorted_list[:22]
                predict_test[i] = sorted_list
            else:
                nmf_number=nmf_number_min

                if len(sorted_list) < 22 - nmf_number:
                    nmf_number = 22 - len(sorted_list)

                est_user_eval = np.dot(user_feature_matrix[id_dic['user_id'].index(i)], item_feature_matrix)
                tmp = sorted(zip(est_user_eval, id_dic['product_id']), key=lambda x: x[0], reverse=True)
                predict = list(zip(*tmp))[1]

                add_list = []
                num = 0
                while len(add_list) != nmf_number:
                    if predict[num] not in sorted_list:
                        add_list.append(predict[num])
                    num += 1
                if len(sorted_list) > 22 - nmf_number:
                    sorted_list = sorted_list[:22 - nmf_number]
                sorted_list.extend(add_list)
                predict_test[i] = sorted_list
        return predict_test

    # 方法12 - 過去とNMFのハイブリッド+randomforest推薦
    def method12_hybrid_forest(self, name, test_ids):
        with open('../data/time_weight/fitting_balanced_' + name + '.pickle', 'rb') as f:
            time_weight = pickle.load(f)
        parm_dic = {'A': {'conv': 0, 'click': 0.20701892, 'view': 0.78720054, 'cart': 0.19557122},
                    'B': {'conv': 1, 'click': 0.43314098, 'view': 0.5480186, 'cart': 1},
                    'C': {'conv': 0, 'click': 0, 'view': 0.71978554, 'cart': 1},
                    'D': {'conv': 1, 'click': 0, 'view': 0.82985685, 'cart': 0}}
        if name != 'D':
            with open('../data/matrix/all_time_weighted_' + name + '.pickle', 'rb') as f:
                sparse_data = pickle.load(f)
            with open('../data/matrix/all_id_dic_time_weighted_' + name + '.pickle', 'rb') as f:
                id_dic = pickle.load(f)
            model = NMF(n_components=128, max_iter=1024, tol=0.001)
            user_feature_matrix = model.fit_transform(sparse_data)
            item_feature_matrix = model.components_
            if name != 'C':
                with open('../data/conv_pred/train_data2_' + name + '.pickle', 'rb') as f:
                    data = pickle.load(f)
                with open('../data/conv_pred/test_X_cut_' + name + '.pickle', 'rb') as f:
                    name_dic_train = pickle.load(f)
                v = DictVectorizer()
                X = v.fit_transform(data['X'])
                y = np.array(data['y'])

                forest = BalancedBaggingClassifier(n_estimators=100, n_jobs=1)
                forest.fit(X, y)
        test_min = datetime.datetime(year=2017, month=5, day=1)
        predict_test = {}
        for i in tqdm.tqdm(test_ids):
            # ユニークitem idを取得
            tmp_dict = {}
            past_items = pd.unique(self.personal_train[name][i]['product_id'])

            # 過去のデータから商品の重みを計算
            for j in past_items:
                tmp_dict[j] = 0
                for _, row in self.personal_train[name][i][
                            self.personal_train[name][i]['product_id'] == j].iterrows():
                    if row['event_type'] == 1:
                        tmp_dict[j] += parm_dic[name]['view'] * time_weight[
                            -1 * (row['time_stamp'] - test_min).days]
                    elif row['event_type'] == 0:
                        tmp_dict[j] += parm_dic[name]['cart'] * time_weight[
                            -1 * (row['time_stamp'] - test_min).days]
                    elif row['event_type'] == 2:
                        tmp_dict[j] += parm_dic[name]['click'] * time_weight[
                            -1 * (row['time_stamp'] - test_min).days]
                    elif row['event_type'] == 3:
                        tmp_dict[j] += parm_dic[name]['conv'] * time_weight[
                            -1 * (row['time_stamp'] - test_min).days]

            sorted_list = sorted(tmp_dict.items(), key=itemgetter(1), reverse=True)
            sorted_list = [x for x, y in sorted_list]
            if name == 'D':
                if len(sorted_list) > 22:
                    sorted_list = sorted_list[:22]
                predict_test[i] = sorted_list
            else:
                if name!='C':
                    sorted_list2 = []
                    input_data = []
                    for k in sorted_list:
                        if k in name_dic_train[i].keys() and len(name_dic_train[i][k]) != 0:
                            sorted_list2.append(k)
                            input_data.append(name_dic_train[i][k])
                    if len(input_data) != 0:
                        X = v.transform(input_data)
                        pred = forest.predict_proba(X)
                        pred = pred[:, 1]
                        conv_list = []
                        mysort = sorted(zip(sorted_list2, pred), key=lambda x: x[1], reverse=True)
                        for k in range(len(mysort)):
                            # if mysort[k][1] >= 0.5:
                            #     conv_list.append(mysort[k][0])
                            conv_list.append(mysort[k][0])
                        # for k in sorted_list:
                        #     if k not in conv_list:
                        #         conv_list.append(k)
                        sorted_list = conv_list
                if len(sorted_list) > 22:
                    sorted_list = sorted_list[:22]
                nmf_number=22-len(sorted_list)

                if nmf_number>0:
                    est_user_eval = np.dot(user_feature_matrix[id_dic['user_id'].index(i)], item_feature_matrix)
                    #est_user_eval = cm.dot(cm.CUDAMatrix(user_feature_matrix[id_dic['user_id'].index(i):id_dic['user_id'].index(i) + 1]),cm.CUDAMatrix(item_feature_matrix)).asarray()[0]
                    tmp = sorted(zip(est_user_eval, id_dic['product_id']), key=lambda x: x[0], reverse=True)
                    predict = list(zip(*tmp))[1]

                    add_list = []
                    num = 0
                    while len(add_list) != nmf_number:
                        if predict[num] not in sorted_list:
                            add_list.append(predict[num])
                        num += 1
                    sorted_list.extend(add_list)
                predict_test[i] = sorted_list
        return predict_test

    # 方法12 - 過去とNMFのハイブリッド+randomforest推薦
    def method13_notRec(self, name, test_ids):
        with open('../data/time_weight/fitting_balanced_' + name + '.pickle', 'rb') as f:
            time_weight = pickle.load(f)
        parm_dic = {'A': {'conv': 0, 'click': 0.20701892, 'view': 0.78720054, 'cart': 0.19557122},
                    'B': {'conv': 1, 'click': 0.43314098, 'view': 0.5480186, 'cart': 1},
                    'C': {'conv': 0, 'click': 0, 'view': 0.71978554, 'cart': 1},
                    'D': {'conv': 1, 'click': 0, 'view': 0.82985685, 'cart': 0}}
        if name != 'D':
            with open('../data/matrix/all_time_weighted_' + name + '.pickle', 'rb') as f:
                sparse_data = pickle.load(f)
            with open('../data/matrix/all_id_dic_time_weighted_' + name + '.pickle', 'rb') as f:
                id_dic = pickle.load(f)
            model = NMF(n_components=128, max_iter=1024, tol=0.001)
            user_feature_matrix = model.fit_transform(sparse_data)
            item_feature_matrix = model.components_
            if name != 'C':
                with open('../data/conv_pred/train_data_' + name + '.pickle', 'rb') as f:
                    data = pickle.load(f)
                with open('../data/conv_pred/test_X_cut_origin_' + name + '.pickle', 'rb') as f:
                    name_dic_train = pickle.load(f)
                v = DictVectorizer()
                X = v.fit_transform(data['X'])
                y = np.array(data['y'])

                forest = BalancedBaggingClassifier(n_estimators=500, n_jobs=1,random_state=777)
                forest.fit(X, y)
                forest2 = BalancedBaggingClassifier(n_estimators=500, n_jobs=1, random_state=1234)
                forest2.fit(X, y)
                forest3 = BalancedBaggingClassifier(n_estimators=500, n_jobs=1, random_state=1919)
                forest3.fit(X, y)
                forest4 = BalancedBaggingClassifier(n_estimators=500, n_jobs=1, random_state=114514)
                forest4.fit(X, y)
                forest5 = BalancedBaggingClassifier(n_estimators=500, n_jobs=1, random_state=334)
                forest5.fit(X, y)
                # with open('../data/conv_pred/train_data_notRec_' + name + '.pickle', 'rb') as f:
                #     data = pickle.load(f)
                # X = v.transform(data['X'])
                # y = np.array(data['y'])
                #
                # notRecforest = BalancedBaggingClassifier(n_estimators=100, n_jobs=1)
                # notRecforest.fit(X, y)
        test_min = datetime.datetime(year=2017, month=5, day=1)
        predict_test = {}
        for i in tqdm.tqdm(test_ids):
            # ユニークitem idを取得
            tmp_dict = {}
            past_items = pd.unique(self.personal_train[name][i]['product_id'])

            # 過去のデータから商品の重みを計算
            for j in past_items:
                tmp_dict[j] = 0
                for _, row in self.personal_train[name][i][
                            self.personal_train[name][i]['product_id'] == j].iterrows():
                    if row['event_type'] == 1:
                        tmp_dict[j] += parm_dic[name]['view'] * time_weight[
                            -1 * (row['time_stamp'] - test_min).days]
                    elif row['event_type'] == 0:
                        tmp_dict[j] += parm_dic[name]['cart'] * time_weight[
                            -1 * (row['time_stamp'] - test_min).days]
                    elif row['event_type'] == 2:
                        tmp_dict[j] += parm_dic[name]['click'] * time_weight[
                            -1 * (row['time_stamp'] - test_min).days]
                    elif row['event_type'] == 3:
                        tmp_dict[j] += parm_dic[name]['conv'] * time_weight[
                            -1 * (row['time_stamp'] - test_min).days]

            sorted_list = sorted(tmp_dict.items(), key=itemgetter(1), reverse=True)
            sorted_list = [x for x, y in sorted_list]
            old_set = sorted_list
            if name == 'D':
                if len(sorted_list) > 22:
                    sorted_list = sorted_list[:22]
                predict_test[i] = sorted_list
            else:
                if name != 'C':
                    sorted_list2 = []
                    input_data = []
                    for k in sorted_list:
                        if k in name_dic_train[i].keys() and len(name_dic_train[i][k]) != 0:
                            sorted_list2.append(k)
                            input_data.append(name_dic_train[i][k])
                    if len(input_data) != 0:
                        X = v.transform(input_data)
                        pred = forest.predict_proba(X)[:,1]
                        pred2 = forest2.predict_proba(X)[:, 1]
                        pred3 = forest3.predict_proba(X)[:, 1]
                        pred4 = forest4.predict_proba(X)[:, 1]
                        pred5 = forest5.predict_proba(X)[:, 1]
                        pred=(pred+pred2+pred3+pred4+pred5)/5
                        #pred_notRec = notRecforest.predict_proba(X)[:,1]
                        conv_list = []
                        rec_list=[]
                        mysort = sorted(zip(sorted_list2, pred), key=lambda x: x[1], reverse=True)
                        #notRecsort = sorted(zip(sorted_list2, pred_notRec), key=lambda x: x[1], reverse=False)
                        # for k in range(len(notRecsort)):
                        #     if notRecsort[k][1] >= 0.5:
                        #         conv_list.append(notRecsort[k][0])
                        for k in range(len(mysort)):
                            if mysort[k][1] >= 0.5:
                                rec_list.append(mysort[k][0])
                        for k in old_set:
                            if k not in rec_list:
                                rec_list.append(k)
                        sorted_list = rec_list
                if len(sorted_list) > 22:
                    sorted_list = sorted_list[:22]
                # elif name == 'A':
                #     for k in conv_list:
                #         if len(sorted_list) >= 22:
                #             break
                #         if k not in sorted_list:
                #             sorted_list.append(k)
                nmf_number = 22 - len(sorted_list)
                if len(sorted_list) > 22:
                    sorted_list = sorted_list[:22]

                if nmf_number > 0:
                    est_user_eval = np.dot(user_feature_matrix[id_dic['user_id'].index(i)], item_feature_matrix)
                    # est_user_eval = cm.dot(cm.CUDAMatrix(user_feature_matrix[id_dic['user_id'].index(i):id_dic['user_id'].index(i) + 1]),cm.CUDAMatrix(item_feature_matrix)).asarray()[0]
                    tmp = sorted(zip(est_user_eval, id_dic['product_id']), key=lambda x: x[0], reverse=True)
                    predict = list(zip(*tmp))[1]

                    add_list = []
                    num = 0
                    while len(add_list) != nmf_number:
                        if predict[num] not in sorted_list:
                            add_list.append(predict[num])
                        num += 1
                    sorted_list.extend(add_list)
                predict_test[i] = sorted_list
        return predict_test

    def all_predict(self):
        print('予測開始します')
        predict_ids={}
        for i in ['A','B','C','D']:
            print(i)
            predict_ids[i]=self.method13_notRec(i, self.submit_ids[i])

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