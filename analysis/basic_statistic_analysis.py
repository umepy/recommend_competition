#coding:utf-8
#データの基本統計量を調べる

import numpy as np
import pandas as pd
from numba import jit
import time
import math
import seaborn as sns
import matplotlib.pyplot as plt
import tqdm
import pickle
import datetime
from scipy import stats
from scipy.sparse import lil_matrix
from sklearn.decomposition import NMF
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from multiprocessing import Process,Manager
from fastFM import mcmc,als
from pyfm import pylibfm
import GPyOpt
from sklearn.preprocessing import Normalizer
import cudamat as cm

# データ読み込み
def read_data(name):
    train = pd.read_csv('../data/train/train_'+name+'.tsv',delimiter='\t',parse_dates=['time_stamp'])
    return train
def read_personal_data(name):
    with open('../data/personal/personal_'+name+'.pickle', 'rb') as f:
        data=pickle.load(f)
    return data
def read_personal_test(name):
    data = pd.read_pickle('../data/personal/personal_test_'+name+'.pickle')
    return data
def read_personal_train(name):
    data=pd.read_pickle('../data/personal/personal_train_'+name+'.pickle')
    return data
def read_personal_test_idcg(name):
    with open('../data/personal/personal_test_items_IDCG_'+name+'.pickle', 'rb') as f:
        data=pickle.load(f)
    return data

# 基本統計量算出
def statistic_analysis(data,name=None):
    st=time.time()
    print(data.columns)
    print('number of uniqe ids: '+str(data.user_id.value_counts().count()))
    print('\nコンバージョン:3, クリック:2, 閲覧:1, カート:0')
    print(data['event_type'].value_counts())
    print(data['event_type'].value_counts()/sum(data['event_type'].value_counts()))
    print('\n')
    print(data['ad'].value_counts())

    if name != None:
        #ユニークIDを取得
        personal_dic = read_personal_data(name)
        num=[]
        for i in personal_dic.keys():
            if len(personal_dic[i]) < 1000:
                num.append(len(personal_dic[i]))
        print('Time : '+str(time.time() - st))
        #sns.distplot(num,kde=False,bins=100)
        #plt.show()
        print('number of category: {0}'.format(len(num)))
        print('mean of persons num: {0}'.format(np.mean(num)))
        print('std of persons num: {0}'.format(np.std(num)))
        print('1Q : {0}'.format(stats.scoreatpercentile(num,25)))
        print('2Q : {0}'.format(stats.scoreatpercentile(num, 50)))
        print('3Q : {0}'.format(stats.scoreatpercentile(num, 75)))
        print('\nテスト期間でのユニーク商品数')

        test_dic=read_personal_test(name)
        num=[]
        for i in test_dic.keys():
            num.append(len(pd.unique(test_dic[i]['product_id'])))
        print('number of category: {0}'.format(len(num)))
        print('mean of persons num: {0}'.format(np.mean(num)))
        print('std of persons num: {0}'.format(np.std(num)))
        print('1Q : {0}'.format(stats.scoreatpercentile(num, 25)))
        print('2Q : {0}'.format(stats.scoreatpercentile(num, 50)))
        print('3Q : {0}'.format(stats.scoreatpercentile(num, 75)))


# 各個人のデータを抽出
def extract_personaldata(name,data):
    unique = data.user_id.unique()
    DataFrameDict = {elem: pd.DataFrame for elem in unique}
    for key in tqdm.tqdm(DataFrameDict.keys()):
        DataFrameDict[key] = data[:][data.user_id == key]
    with open('../data/personal/personal_'+name+'.pickle','wb') as f:
        pickle.dump(DataFrameDict,f)

# 各アイテムのデータを抽出
def extract_itemdata(name):
    data=read_data(name)
    unique = data.product_id.unique()
    DataFrameDict = {elem: pd.DataFrame for elem in unique}
    for key in tqdm.tqdm(DataFrameDict.keys()):
        DataFrameDict[key] = data[:][data.product_id == key]
    with open('../data/personal/item_' + name + '.pickle', 'wb') as f:
        pickle.dump(DataFrameDict, f)

# 個人のデータをA,B,C,D全てで抽出
def extract_all():
    for i in ('A','B','C','D'):
        a=read_data(i)
        extract_personaldata(i,a)
        extract_ranking(i)
    get_predict_ids()
    view_time_fitting(True)

# 各個人のテスト期間での商品上位とIDCGの算出
def extract_ranking(name):
    # 個人のデータ読み込み
    with open('../data/personal/personal_' + name + '.pickle', 'rb') as f:
        df = pickle.load(f)

    out_dic=dict()
    test_dic=dict()
    train_dic=dict()
    #各ユーザに対して商品の関連度とIDCGの辞書を作成し，user_idをキーとした辞書を作成
    for i in tqdm.tqdm(df.keys()):
        tmp_dic,tmp_test,tmp_train=extract_items(df[i])
        out_dic[i]=tmp_dic
        test_dic[i]=tmp_test
        train_dic[i]=tmp_train

    with open('../data/personal/personal_test_items_IDCG_' + name + '.pickle','wb') as f:
        pickle.dump(out_dic,f)
    with open('../data/personal/personal_test_' + name + '.pickle', 'wb') as f:
        pickle.dump(test_dic, f)
    with open('../data/personal/personal_train_' + name + '.pickle','wb') as f:
        pickle.dump(train_dic,f)

# 個人のデータから商品idと関連度を算出
def extract_items(data):
    test=data[data['time_stamp'] > datetime.datetime(year=2017, month=4, day=24)]
    train = data[data['time_stamp'] <= datetime.datetime(year=2017, month=4, day=24)]
    out_dic=dict()

    #テスト期間でのidの取得
    product_ids=pd.unique(test['product_id'])
    for id in product_ids:
        tmp_df=test[test['product_id']==id].reset_index()
        out_dic[id] = 0
        for j in range(len(tmp_df)):
            if out_dic[id] < tmp_df['event_type'][j] and tmp_df['ad'][j] != 0:
                out_dic[id] = tmp_df['event_type'][j]

    #IDCGの計算
    scores=list(out_dic.values())
    scores.reverse()
    out_dic['IDCG']=calc_IDCG(scores)
    return out_dic,test,train

# IDCGの計算
def calc_IDCG(rank):
    idcg=0
    if len(rank)>=22:
        for i in range(22):
            idcg+=(2**rank[i]-1)/np.log2((i+1)+1)
    else:
        for i in range(len(rank)):
            idcg += (2 ** rank[i] - 1) / np.log2((i + 1) + 1)
    return idcg

# 予測用idの取得
def get_predict_ids():
    df=pd.read_csv('../data/sample_submit.tsv',delimiter='\t',names=['user_id','item_id','rank'])
    ids=pd.unique(df['user_id'])
    ids_dic={}
    tmp_A=[]
    tmp_B=[]
    tmp_C=[]
    tmp_D=[]
    for i in ids:
        if 'A' in i:
            tmp_A.append(i)
        elif 'B' in i:
            tmp_B.append(i)
        elif 'C' in i:
            tmp_C.append(i)
        else:
            tmp_D.append(i)
    ids_dic['A']=tmp_A
    ids_dic['B']=tmp_B
    ids_dic['C']=tmp_C
    ids_dic['D']=tmp_D

    with open('../data/submit_ids.pickle','wb') as f:
        pickle.dump(ids_dic,f)

# テスト期間における訓練期間のitemの含有量
def check_persentage_of_items_in_test():
    for name in ['A','B','C','D']:
        train=read_personal_train(name)
        test=read_personal_test(name)
        idcg=read_personal_test_idcg(name)

        # 各種スコアの初期化
        all_items=0
        all_count=0
        conv_items=0
        conv_count=0
        conv_ad=0
        click_items=0
        click_count=0
        click_only=0
        view_items=0
        view_count=0
        view_only=0
        top_22=0
        top_count=0

        # テスト期間のユニークIDを取得
        unique_ids = test.keys()
        for i in tqdm.tqdm(unique_ids):
            tmp_train=train[i]
            tmp_test=test[i]
            set_train = set(pd.unique(tmp_train['product_id']))

            set_conv = set()
            set_ad = set()
            set_onlyclick = set()
            set_onlyview=set()

            if len(pd.unique(tmp_test['product_id'])) !=0:
                set_test=set(pd.unique(tmp_test['product_id']))
                set_and=set_train & set_test
                if len(set_and) != 0:
                    all_items+=1.0/len(set_test)*len(set_and)
                all_count+=1

            if len(pd.unique(tmp_test[tmp_test['event_type']==3]['product_id'])) !=0:
                filtered_test=tmp_test[tmp_test['event_type'] == 3]
                set_test=set(pd.unique(filtered_test['product_id']))
                set_conv=set_test
                set_ad=set(pd.unique(filtered_test[filtered_test['ad']==1]['product_id']))
                set_nonad=set_conv-set_ad
                set_and=set_train & set_test
                set_and_ad=set_train & set_ad
                if len(set_and) != 0:
                    conv_items+=1.0/len(set_test)*len(set_and)
                if len(set_and_ad) != 0:
                    conv_ad+=1.0/len(set_ad)*len(set_and_ad)
                conv_count+=1

            if len(pd.unique(tmp_test[tmp_test['event_type']==2]['product_id'])) !=0:
                set_test=set(pd.unique(tmp_test[tmp_test['event_type']==2]['product_id']))
                set_onlyclick=set_test - set_ad
                set_and=set_train & set_test
                set_and_onlyclick=set_train & set_onlyclick
                if len(set_and) != 0:
                    click_items+=1.0/len(set_test)*len(set_and)
                if len(set_and_onlyclick) != 0:
                    click_only+=1.0/len(set_onlyclick)*len(set_and_onlyclick)
                click_count+=1

            if len(pd.unique(tmp_test[tmp_test['event_type']==1]['product_id'])) !=0:
                set_test=set(pd.unique(tmp_test[tmp_test['event_type']==1]['product_id']))
                set_onlyview=set_test - set_ad - set_onlyclick
                set_and=set_train & set_test
                set_and_onlyview=set_train & set_onlyview
                if len(set_and) != 0:
                    view_items+=1.0/len(set_test)*len(set_and)
                if len(set_and_onlyview) != 0:
                    view_only+=1.0/len(set_onlyview)*len(set_and_onlyview)
                view_count+=1

            tmp_idcg=idcg[i]
            del tmp_idcg['IDCG']
            print(tmp_test)
            print('------------------------------------')
            print(tmp_idcg)

            if len(tmp_idcg)!=0:
                set_22=set(list(tmp_idcg.keys()))
                set_and=set_train & set_22
                top_22+=1.0*len(set_and)/len(set_22)
                top_count+=1


        if all_count!=0:
            all_items/=all_count
        if conv_count != 0:
            conv_items/=conv_count
            conv_ad/=conv_count
        if click_count != 0:
            click_items/=click_count
            click_only/=click_count
        if view_count != 0:
            view_items/=view_count
            view_only/=view_count
        if top_count!=0:
            top_22/=top_count

        print('Percentage of '+name)
        print('All items : {:.2%}'.format(all_items))
        print('Conv items : {:.2%}'.format(conv_items))
        print('Conv items(ad only) : {:.2%}'.format(conv_ad))
        print('Click items : {:.2%}'.format(click_items))
        print('Click items(highest) : {:.2%}'.format(click_only))
        print('View items : {:.2%}'.format(view_items))
        print('View items(highest) : {:.2%}'.format(view_only))
        print('top22 items : {:.2%}'.format(top_22))

# 訓練期間の評価値行列を作成(コンバージョンのみ)
def create_evaluate_matrix_conversion(name):
    train_data=read_data(name)
    # 訓練期間のデータをフィルタ
    train_data=train_data[train_data['time_stamp'] <= datetime.datetime(year=2017, month=4, day=24)]
    personal_data=read_personal_train(name)

    unique_product_ids = list(pd.unique(train_data['product_id']))
    unique_user_ids = list(pd.unique(train_data['user_id']))

    print(len(unique_product_ids))
    print(len(unique_user_ids))


    ev_matrix=lil_matrix((len(unique_user_ids),len(unique_product_ids)))
    for user_id in tqdm.tqdm(unique_user_ids):
        for p_id in pd.unique(personal_data[user_id]['product_id']):
            ev_matrix[unique_user_ids.index(user_id),unique_product_ids.index(p_id)]=len(personal_data[user_id][personal_data[user_id]['event_type']==3])

    save_dic={'user_id':unique_user_ids,'product_id':unique_product_ids}

    with open('../data/matrix/train_only_conversion_'+name+'.pickle', 'wb') as f:
        pickle.dump(ev_matrix, f)
    with open('../data/matrix/id_dic_only_conversion_'+name+'.pickle', 'wb') as f:
        pickle.dump(save_dic, f)

# 訓練期間の評価値行列を作成(重み付け)
def create_evaluate_matrix_weighted(name):
    train_data=read_data(name)
    # 訓練期間のデータをフィルタ
    train_data=train_data[train_data['time_stamp'] <= datetime.datetime(year=2017, month=4, day=24)]
    personal_data=read_personal_train(name)

    unique_product_ids = list(pd.unique(train_data['product_id']))
    unique_user_ids = list(pd.unique(train_data['user_id']))

    ev_matrix=lil_matrix((len(unique_user_ids),len(unique_product_ids)))
    for user_id in tqdm.tqdm(unique_user_ids):
        for p_id in pd.unique(personal_data[user_id]['product_id']):
            for k in personal_data[user_id][personal_data[user_id]['product_id'] == p_id]['event_type']:
                if k==1:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += 3
                elif k==0:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += 2
                elif k==2:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += 1
                print(k)

    save_dic={'user_id':unique_user_ids,'product_id':unique_product_ids}

    with open('../data/matrix/train_weighted_'+name+'.pickle', 'wb') as f:
        pickle.dump(ev_matrix, f)
    with open('../data/matrix/id_dic_weighted_'+name+'.pickle', 'wb') as f:
        pickle.dump(save_dic, f)

# 訓練期間の評価値行列を作成(時間重み付け)
def create_evaluate_matrix_time_weighted(name):
    train_data = read_data(name)
    # 時間減衰を読み込み
    with open('../data/time_weight/fitting_balanced_' + name + '.pickle', 'rb') as f:
        time_weight = pickle.load(f)
    test_min=datetime.datetime(year=2017, month=4, day=24)
    # 訓練期間のデータをフィルタ
    train_data = train_data[train_data['time_stamp'] <= datetime.datetime(year=2017, month=4, day=24)]
    personal_data = read_personal_train(name)

    unique_product_ids = list(pd.unique(train_data['product_id']))
    unique_user_ids = list(pd.unique(train_data['user_id']))

    ev_matrix = lil_matrix((len(unique_user_ids), len(unique_product_ids)))
    for user_id in tqdm.tqdm(unique_user_ids):
        for p_id in pd.unique(personal_data[user_id]['product_id']):
            for _, row in personal_data[user_id][personal_data[user_id]['product_id'] == p_id].iterrows():
                if row['event_type'] == 1:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += 3 * time_weight[-1 * (row['time_stamp'] - test_min).days]
                elif row['event_type'] == 0:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += 2 * time_weight[-1 * (row['time_stamp'] - test_min).days]
                elif row['event_type'] == 2:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += 1 * time_weight[-1 * (row['time_stamp'] - test_min).days]

    save_dic = {'user_id': unique_user_ids, 'product_id': unique_product_ids}

    with open('../data/matrix/train_time_weighted_' + name + '.pickle', 'wb') as f:
        pickle.dump(ev_matrix, f)
    with open('../data/matrix/id_dic_time_weighted_' + name + '.pickle', 'wb') as f:
        pickle.dump(save_dic, f)

    # 訓練期間の評価値行列を作成()
def create_evaluate_matrix_optimize(name):
    train_data = read_data(name)
    # 時間減衰を読み込み
    with open('../data/time_weight/fitting_balanced_' + name + '.pickle', 'rb') as f:
        time_weight = pickle.load(f)
    parm_dic = {'A': {'conv': 0, 'click': 0.20701892, 'view': 0.78720054, 'cart': 0.19557122},
                'B': {'conv': 1, 'click': 0.43314098, 'view': 0.5480186, 'cart': 1},
                'C': {'conv': 0, 'click': 0, 'view': 0.71978554, 'cart': 1},
                'D': {'conv': 1, 'click': 0, 'view': 0.82985685, 'cart': 0}}
    test_min = datetime.datetime(year=2017, month=4, day=24)
    # 訓練期間のデータをフィルタ
    train_data = train_data[train_data['time_stamp'] <= datetime.datetime(year=2017, month=4, day=24)]
    personal_data = read_personal_train(name)

    unique_product_ids = list(pd.unique(train_data['product_id']))
    unique_user_ids = list(pd.unique(train_data['user_id']))

    ev_matrix = lil_matrix((len(unique_user_ids), len(unique_product_ids)))
    for user_id in tqdm.tqdm(unique_user_ids):
        for p_id in pd.unique(personal_data[user_id]['product_id']):
            for _, row in personal_data[user_id][personal_data[user_id]['product_id'] == p_id].iterrows():
                if row['event_type'] == 1:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += parm_dic[name]['view'] * time_weight[
                        -1 * (row['time_stamp'] - test_min).days]
                elif row['event_type'] == 0:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += parm_dic[name]['cart'] * time_weight[
                        -1 * (row['time_stamp'] - test_min).days]
                elif row['event_type'] == 2:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += parm_dic[name]['click'] * time_weight[
                        -1 * (row['time_stamp'] - test_min).days]
                elif row['event_type'] == 3:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += parm_dic[name]['conv'] * time_weight[
                        -1 * (row['time_stamp'] - test_min).days]

    save_dic = {'user_id': unique_user_ids, 'product_id': unique_product_ids}

    with open('../data/matrix/train_optimized_' + name + '.pickle', 'wb') as f:
        pickle.dump(ev_matrix, f)
    with open('../data/matrix/id_dic_optimized_' + name + '.pickle', 'wb') as f:
        pickle.dump(save_dic, f)

# 全期間の評価値行列を作成(重み付け)
def all_evaluate_matrix_weighted(name):
    train_data=read_data(name)
    personal_data=read_personal_train(name)

    unique_product_ids = list(pd.unique(train_data['product_id']))
    unique_user_ids = list(pd.unique(train_data['user_id']))

    ev_matrix=lil_matrix((len(unique_user_ids),len(unique_product_ids)))
    for user_id in tqdm.tqdm(unique_user_ids):
        for p_id in pd.unique(personal_data[user_id]['product_id']):
            for k in personal_data[user_id][personal_data[user_id]['product_id'] == p_id]['event_type']:
                if k==1:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += 3
                elif k==0:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += 2
                elif k==2:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += 1
                print(k)

    save_dic={'user_id':unique_user_ids,'product_id':unique_product_ids}

    with open('../data/matrix/all_weighted_'+name+'.pickle', 'wb') as f:
        pickle.dump(ev_matrix, f)
    with open('../data/matrix/all_id_dic_weighted_'+name+'.pickle', 'wb') as f:
        pickle.dump(save_dic, f)

# 全期間の評価値行列を作成(重み付け)
def all_evaluate_matrix_time_weighted(name):
    train_data = read_data(name)
    # 時間減衰を読み込み
    with open('../data/time_weight/fitting_balanced_' + name + '.pickle', 'rb') as f:
        time_weight = pickle.load(f)
    test_min = datetime.datetime(year=2017, month=5, day=1)
    personal_data = read_personal_train(name)

    unique_product_ids = list(pd.unique(train_data['product_id']))
    unique_user_ids = list(pd.unique(train_data['user_id']))

    ev_matrix = lil_matrix((len(unique_user_ids), len(unique_product_ids)))
    for user_id in tqdm.tqdm(unique_user_ids):
        for p_id in pd.unique(personal_data[user_id]['product_id']):
            for _, row in personal_data[user_id][personal_data[user_id]['product_id'] == p_id].iterrows():
                if row['event_type'] == 1:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += 3 * time_weight[
                        -1 * (row['time_stamp'] - test_min).days]
                elif row['event_type'] == 0:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += 2 * time_weight[
                        -1 * (row['time_stamp'] - test_min).days]
                elif row['event_type'] == 2:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += 1 * time_weight[
                        -1 * (row['time_stamp'] - test_min).days]

    save_dic = {'user_id': unique_user_ids, 'product_id': unique_product_ids}
    with open('../data/matrix/all_time_weighted_' + name + '.pickle', 'wb') as f:
        pickle.dump(ev_matrix, f)
    with open('../data/matrix/all_id_dic_time_weighted_' + name + '.pickle', 'wb') as f:
        pickle.dump(save_dic, f)

# 全期間の評価値行列を作成(重み付け)
def all_evaluate_matrix_optimized(name):
    train_data = read_data(name)
    # 時間減衰を読み込み
    with open('../data/time_weight/fitting_balanced_' + name + '.pickle', 'rb') as f:
        time_weight = pickle.load(f)
    parm_dic = {'A': {'conv': 0, 'click': 0.20701892, 'view': 0.78720054, 'cart': 0.19557122},
                'B': {'conv': 1, 'click': 0.43314098, 'view': 0.5480186, 'cart': 1},
                'C': {'conv': 0, 'click': 0, 'view': 0.71978554, 'cart': 1},
                'D': {'conv': 1, 'click': 0, 'view': 0.82985685, 'cart': 0}}
    test_min = datetime.datetime(year=2017, month=5, day=1)
    personal_data = read_personal_train(name)

    unique_product_ids = list(pd.unique(train_data['product_id']))
    unique_user_ids = list(pd.unique(train_data['user_id']))

    ev_matrix = lil_matrix((len(unique_user_ids), len(unique_product_ids)))
    for user_id in tqdm.tqdm(unique_user_ids):
        for p_id in pd.unique(personal_data[user_id]['product_id']):
            for _, row in personal_data[user_id][personal_data[user_id]['product_id'] == p_id].iterrows():
                if row['event_type'] == 1:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += parm_dic[name]['view'] * time_weight[
                        -1 * (row['time_stamp'] - test_min).days]
                elif row['event_type'] == 0:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += parm_dic[name]['cart'] * time_weight[
                        -1 * (row['time_stamp'] - test_min).days]
                elif row['event_type'] == 2:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += parm_dic[name]['click'] * time_weight[
                        -1 * (row['time_stamp'] - test_min).days]
                elif row['event_type'] == 3:
                    ev_matrix[unique_user_ids.index(user_id), unique_product_ids.index(p_id)] += parm_dic[name]['conv'] * time_weight[
                        -1 * (row['time_stamp'] - test_min).days]

    save_dic = {'user_id': unique_user_ids, 'product_id': unique_product_ids}
    with open('../data/matrix/all_optimized_' + name + '.pickle', 'wb') as f:
        pickle.dump(ev_matrix, f)
    with open('../data/matrix/all_id_dic_optimized_' + name + '.pickle', 'wb') as f:
        pickle.dump(save_dic, f)

# -----時間と推薦商品の関係の可視化-----
def extract_time_and_past_items():
    for name in ['A','B','C','D']:
        train=read_personal_train(name)
        test=read_personal_test(name)
        idcg=read_personal_test_idcg(name)

        # トレイン期間とテスト期間の商品の時間の差分
        days_all=Manager().list()
        days_and=Manager().list()

        jobs=[]
        n=len(idcg.keys())//8
        users=list(zip(*[iter(list(idcg.keys()))] * n))
        for i in range(8):
            p = Process(target=extract_sub, args=(users[i], train,test,days_all,days_and))
            jobs.append(p)
        [x.start() for x in jobs]
        [x.join() for x in jobs]
        days_all=list(days_all)
        days_and=list(days_and)

        if len(days_all)==0:
            continue

        out_dic={'all':days_all,'and':days_and}
        with open('../data/view/dic_number_cart_'+name+'_.pickle','wb') as f:
            pickle.dump(out_dic,f)

def extract_sub(users,train,test,d_all,d_and):
    # テスト開始期間に合わせる
    test_min = datetime.datetime(year=2017, month=4, day=24)
    # 各ユーザに対して
    for user in tqdm.tqdm(users):
        train_unique_items = pd.unique(train[user]['product_id'])
        test_unique_items = pd.unique(test[user]['product_id'])
        # どちらにも存在するitemを抽出
        item_subset = set(train_unique_items) & set(test_unique_items)

        # train期間におけるイベントの期間分布を調べる
        item_subset_all = train_unique_items

        for item in item_subset_all:
            add_flag = False
            if item in item_subset:
                add_flag = True
            # トレイン期間の全イベントのndarrayを作成
            tmp_train = train[user][train[user]['product_id'] == item]
            train_ndarray = pd.to_datetime(tmp_train[tmp_train['event_type'] == 0]['time_stamp']) - test_min
            for i in train_ndarray:
                d_all.append(i.days)
                if add_flag:
                    d_and.append(i.days)
def view_time_category(name):
    for categ in ['conv','click','view','cart']:
        with open('../data/view/dic_number_'+categ+'_' + name + '_.pickle', 'rb') as f:
            dic=pickle.load(f)
        days_all=dic['all']
        days_and=dic['and']
        data_all = -1 * np.array(days_all)
        data_and = -1 * np.array(days_and)
        x = list(range(0,31))
        y_all = np.zeros(31)
        y_and = np.zeros(31)
        y_balanced = np.zeros(31)
        for i in data_all:
            y_all[i]+=1
        for i in data_and:
            y_and[i] += 1
        y_all = np.array(y_all) / max(y_all)
        y_and = np.array(y_and) / max(y_and)
        for i in x:
            if y_all[i]!=0:
                y_balanced[i]=y_and[i]/y_all[i]
        y_balanced=np.array(y_balanced)/max(y_balanced)
        regr = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ])
        x=[]
        y=[]
        for i in range(30):
            if i>=len(y_balanced):
                continue
            if y_balanced[i]!=0:
                x.append(i)
                y.append(y_balanced[i])
        regr.fit(np.array(x).reshape((len(x), 1)), np.array(y).reshape((len(x), 1)))
        # make predictions

        plt.plot(x, y, 'o', label='Plot')
        xt = np.linspace(1, 30.0, num=100).reshape((100, 1))
        yt = regr.predict(xt)

        plt.plot(xt, yt, label='Fitting')
        plt.xlabel('days')
        plt.ylabel('amount')
        plt.title('Distribution of ' + name+'_'+categ)
        plt.show()

def view_time_fitting(balance):
    for name in ['A', 'B', 'C', 'D']:
        with open('../data/view/time_teststart_' + str(name) + '.pickle', 'rb') as f:
            data=pickle.load(f)
        with open('../data/view/time_all_train_teststart_' + str(name) + '.pickle', 'rb') as f:
            all_data = pickle.load(f)
        data=-1*np.array(data)
        all_data = -1 * np.array(all_data)
        x=[]
        y=[]
        all_y=[]
        for i in data:
            if i not in x:
                x.append(i)
                y.append(1)
                all_y.append(0)
            else:
                y[x.index(i)]+=1
        if balance:
            for i in all_data:
                if i in x:
                    all_y[x.index(i)] += 1
            all_y=np.array(all_y)/max(all_y)
            for i in range(len(y)):
                y[i]/=all_y[i]
        y=np.array(y)/max(y)
        # train a linear regression model
        regr = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ])
        regr.fit(np.array(x).reshape((24,1)), np.array(y).reshape((24,1)))

        # make predictions
        xt = np.linspace(0.0, 30.0, num=100).reshape((100,1))
        yt = regr.predict(xt)

        # 重み曲線の出力
        if balance:
            days=[]
            for i in range(35):
                days.append(i)
            a=regr.predict(np.array(days).reshape((35,1)))
            output=[]
            for i in a:
                if i > 1:
                    output.append(1)
                elif i <0.1:
                    output.append(0.1)
                else:
                    output.append(i[0])
            with open('../data/time_weight/fitting_balanced_' + name + '.pickle','wb') as f:
                pickle.dump(output,f)

        # plot samples and regression result
        plt.plot(x, y, 'o',label='Plot')
        plt.plot(xt, yt,label='Fitting')
        plt.title('Category_'+name)
        plt.legend()
        plt.xlabel('Days')
        plt.ylabel('Importance')
        if balance:
            plt.savefig('../data/view/fitting_balanced_' + name + '.png')
        else:
            plt.savefig('../data/view/fitting_'+name+'.png')
        sns.distplot(data)
        plt.show()
def view_time():
    for name in ['A', 'B', 'C', 'D']:
        with open('../data/view/time_teststart_' + str(name) + '.pickle', 'rb') as f:
            test_number = pickle.load(f)
        with open('../data/view/time_all_train_teststart_' + str(name) + '.pickle', 'rb') as f:
            data=pickle.load(f)
        test_number = -1 * np.array(test_number)
        data=-1*np.array(data)
        x=[]
        y=[]
        for i in data:
            if i not in x:
                x.append(i)
                y.append(1)
            else:
                y[x.index(i)]+=1
        y=np.array(y)/max(y)
        sns.distplot(data)
        plt.xlabel('days')
        plt.ylabel('amount')
        plt.title('Distribution of '+name)
        plt.savefig('../data/view/time_all_train_' + str(name) + '.png')
        plt.show()
        plt.xlabel('days')
        plt.ylabel('amount')
        plt.title('Distribution of ' + name)
        sns.distplot(test_number)
        plt.savefig('../data/view/time_train_' + str(name) + '.png')
        plt.show()

def connect_content():
    for name in ['A','B','C','D']:
        output = {}
        for i in range(1,6):
            with open('../data/view/analysis_content_'+name+str(i)+'.pickle','rb') as f:
                data=pickle.load(f)
            output.update(data)
        with open('../data/view/analysis_content_' + name+'.pickle', 'wb') as f:
            pickle.dump(output,f)

# -----含有率の分析-----
def analysis_content(view):
    if not view:
        for name in ['A', 'B', 'C', 'D']:
            with open('../data/personal/personal_test_items_IDCG_' + name + '.pickle', 'rb') as f:
                IDCG = pickle.load(f)
            with open('../data/view/analysis_content_' + name + '.pickle', 'rb') as f:
                data=pickle.load(f)
            #items, importance
            #もし入っていたらリストに重度度追加
            output=[]
            # 0.01刻みで10-0の範囲
            division_true = np.zeros(1000)
            division_false = np.zeros(1000)
            for user in tqdm.tqdm(data.keys()):
                importance = np.array(data[user]['importance'])

                if len(importance)==0:
                    continue
                #importance = (importance - np.mean(importance))/np.std(importance)
                for i in range(len(data[user]['items'])):
                    if data[user]['items'][i] in IDCG[user]:
                        if np.isnan(importance[i])==False:
                            tmp = math.floor(importance[i] )
                            if tmp >= 1000:
                                tmp = 999
                            if tmp < 0:
                                tmp = 0
                            division_true[tmp]+=1
                            output.append(importance[i])
                    else:
                        if np.isnan(importance[i]) == False:
                            tmp=math.floor(importance[i])
                            if tmp>=1000:
                                tmp=999
                            if tmp<0:
                                tmp=0
                            division_false[tmp] += 1
            # 逆順にして累積和をとる
            division_true = np.cumsum(division_true[::-1])[::-1]
            division_false = np.cumsum(division_false[::-1])[::-1]
            div_final=division_true/(division_true+division_false)*100
            with open('../data/view/importance_' + name + '.pickle', 'wb') as f:
                pickle.dump(div_final,f)
            with open('../data/view/importance_distribution_'+name+'.pickle','wb') as f:
                pickle.dump(output,f)
    else:
        for name in ['A', 'B', 'C', 'D']:
            with open('../data/view/importance_' + name + '.pickle', 'rb') as f:
                data=pickle.load(f)
            x=np.linspace(0,1000,1000)
            plt.plot(x,data)
            plt.title('A relationship between including percentage and importance of '+name)
            plt.xlabel('importance')
            plt.ylabel('including percentage')
            plt.tight_layout()
            plt.savefig('../data/view/importance_including_rate_'+name+'.png')
            plt.show()

# 各要素数のNMFの結果を保存する関数
def nmf_save(components):
    print('component=' + str(components))
    for name in ['A','B','C']:
        with open('../data/matrix/train_time_weighted_' + name + '.pickle', 'rb') as f:
            sparse_data = pickle.load(f)
        model=NMF(n_components=components,verbose=True)
        user=model.fit_transform(sparse_data)
        item=model.components_
        print('component='+str(components))
        with open('../data/nmf/nmf_user_'+str(components)+'_'+name+'.pickle','wb') as f:
            pickle.dump(user,f)
        with open('../data/nmf/nmf_item_'+str(components)+'_'+name+'.pickle','wb') as f:
            pickle.dump(item,f)

# NMFが推薦した商品の含有率を調べる
def nmf_analysis(name,components):
    # GPUを用いて行列演算（初期化）
    cm.cuda_set_device(0)
    cm.init()

    with open('../data/nmf/nmf_user_' + str(components) + '_' + name + '.pickle', 'rb') as f:
        user_feature=pickle.load(f)
    with open('../data/nmf/nmf_item_' + str(components) + '_' + name + '.pickle', 'rb') as f:
        item_feature=pickle.load(f)
    with open('../data/matrix/train_time_weighted_' + name + '.pickle', 'rb') as f:
        sparse_data = pickle.load(f).tocsc()
    with open('../data/matrix/id_dic_time_weighted_' + name + '.pickle', 'rb') as f:
        id_dic = pickle.load(f)
    nmf_result=cm.dot(cm.CUDAMatrix(user_feature),cm.CUDAMatrix(item_feature)).asarray()
    for user in tqdm.tqdm(id_dic['user_id']):
        index_user=id_dic['user_id'].index(user)
        #est_user_eval=np.dot(user_feature[id_dic['user_id'].index(user)],item_feature)
        est_user_eval=nmf_result[index_user]
        tmp = sorted(zip(est_user_eval, id_dic['product_id']), key=lambda x: x[0], reverse=True)
        predict = list(zip(*tmp))[1]
        out_list=[]
        for i in predict:
            index_item = id_dic['product_id'].index(i)
            if sparse_data[index_user,index_item]==0:
                out_list.append(i)

# FM用のデータの作成
def fm_datacreate(name):
    personal_data=read_personal_train(name)
    personal_test=read_personal_test(name)
    with open('../data/time_weight/fitting_balanced_' + name + '.pickle', 'rb') as f:
        time_weight = pickle.load(f)
    manager=Manager()
    rt_data=manager.list()
    # in each user
    split_num=int(len(personal_data)/8)
    jobs=[]
    for i in range(8):
        p = Process(target=fm_multijob, args=(personal_data,personal_test,list(personal_data.keys())[i*split_num:(i+1)*split_num],rt_data,time_weight))
        jobs.append(p)
    [x.start() for x in jobs]
    [x.join() for x in jobs]

    tdata=list(rt_data)
    X = []
    y = []
    for i in tdata:
        X.extend(i['X'])
        y.extend(i['y'])
    with open('../data/ml/ml_real_train__X_' + name + '.pickle', 'wb') as f:
        pickle.dump(X, f)
    with open('../data/ml/ml_real_train_y_' + name + '.pickle', 'wb') as f:
        pickle.dump(y, f)
def fm_multijob(data,test,keys,rt_data,time_weight):
    X=[]
    y=[]
    test_min=datetime.datetime(year=2017,month=4,day=19)
    for user in tqdm.tqdm(keys):
        tmp_train = data[user]
        tmp_test = test[user]
        if len(tmp_train)==0:
            continue
        for item in pd.unique(tmp_train['product_id']):
            user_dic = {'user': user,'item':item,'cart':0,'view':0,'click':0,'conv':0,'first_day':0,'last_day':30,'event_num':0}
            for _, row in tmp_train[tmp_train['product_id'] == item].iterrows():
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

            max_ev=0
            for _, row in tmp_test[tmp_test['product_id'] == item].iterrows():
                if row['event_type'] == 3 and row['ad'] == 1:
                    max_ev=8
                    break
                elif row['event_type'] == 2 and max_ev<4:
                    max_ev=4
                elif row['event_type'] == 1 and max_ev<2:
                    max_ev=2
            X.append(user_dic)
            y.append(max_ev)
    rt_tmp={'X':X,'y':y}
    rt_data.append(rt_tmp)
def connect_fm_train():
    for name in ['A','B','C','D']:
        with open('../data/fm/ev_data_'+name+'.pickle','rb') as f:
            data=pickle.load(f)
        X=[]
        y=[]
        for i in data:
            X.extend(i['X'])
            y.extend(i['y'])
        with open('../data/fm/fm_train_X_'+name+'.pickle','wb') as f:
            pickle.dump(X,f)
        with open('../data/fm/fm_train_y_'+name+'.pickle','wb') as f:
            pickle.dump(y,f)

# 機械学習用のデータの作成
def ml_datacreate():
    for name in ['A','B','C','D']:
        print(name)
        with open('../data/personal/personal_train_'+name+'.pickle','rb') as f:
            personal_data=pickle.load(f)
        with open('../data/time_weight/fitting_balanced_' + name + '.pickle', 'rb') as f:
            time_weight = pickle.load(f)
        manager=Manager()
        rt_data=manager.list()
        # in each user
        split_num=int(len(personal_data)/8)
        jobs=[]
        for i in range(8):
            p = Process(target=ml_multijob, args=(personal_data,list(personal_data.keys())[i*split_num:(i+1)*split_num],rt_data,time_weight))
            jobs.append(p)
        [x.start() for x in jobs]
        [x.join() for x in jobs]

        tdata=list(rt_data)

        X = []
        y = []
        for i in tdata:
            X.extend(i['X'])
            y.extend(i['y'])
        with open('../data/ml/ml_train_onlyone_X_' + name + '.pickle', 'wb') as f:
            pickle.dump(X, f)
        with open('../data/ml/ml_train_onlyone_y_' + name + '.pickle', 'wb') as f:
            pickle.dump(y, f)
def ml_multijob(data,keys,rt_data,time_weight):
    X=[]
    y=[]
    test_min=datetime.datetime(year=2017,month=4,day=19)
    for user in tqdm.tqdm(keys):
        tmp_train = data[user][data[user]['time_stamp'] < datetime.datetime(year=2017, month=4, day=19)]
        tmp_test = data[user][data[user]['time_stamp'] > datetime.datetime(year=2017, month=4, day=19)]
        if len(tmp_train)==0:
            continue
        for item in pd.unique(tmp_train['product_id']):
            user_dic = {'user': user,'item':item,'cart':0,'view':0,'click':0,'conv':0,'first_day':0,'last_day':30,'event_num':0}
            for _, row in tmp_train[tmp_train['product_id'] == item].iterrows():
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

            max_ev=0
            for _, row in tmp_test[tmp_test['product_id'] == item].iterrows():
                if row['event_type'] == 3:
                    max_ev=1
                    break
                elif row['event_type'] == 2 and max_ev<4:
                    max_ev=1
                    break
                elif row['event_type'] == 1 and max_ev<2:
                    max_ev=1
                    break

            # ML用のデータ作成
            ml_data=[user_dic['cart'],user_dic['view'],user_dic['click'],user_dic['conv'],user_dic['first_day'],user_dic['last_day'],user_dic['event_num']]

            X.append(ml_data)
            y.append(max_ev)
    rt_tmp={'X':X,'y':y}
    rt_data.append(rt_tmp)

def fm_test(x):
    print(x)
    w, p_v,l2, iter, rank = float(x[:, 0]), float(x[:, 1]), float(x[:, 2]), int(x[:, 3]),int(x[:, 4])
    name='B'
    with open('../data/fm/fm_train_X_' + name + '.pickle', 'rb') as f:
        r_X=pickle.load(f)
    with open('../data/fm/fm_train_y_' + name + '.pickle', 'rb') as f:
        r_y=pickle.load(f)
    with open('../data/fm/ml_real_train__X_' + name + '.pickle', 'rb') as f:
        test = pickle.load(f)

    v=DictVectorizer()
    X=v.fit_transform(r_X)
    scaler = Normalizer()
    X = scaler.fit_transform(X)
    y=np.array(r_y)

    v_test=v.transform(test)
    fm = als.FMRegression(n_iter=iter,rank=rank,l2_reg_w=w,l2_reg_V=p_v,l2_reg=l2)
    fm.fit(X, y)
    result=fm.predict(v_test)

    output = {}
    user_dic = {}
    for i in range(len(test)):
        if test[i]['user'] not in user_dic:
            user_dic[test[i]['user']]={test[i]['item']:result[i]}
        else:
            user_dic[test[i]['user']][test[i]['item']]=result[i]

    for user in user_dic:
        recommend = [k for k, v in sorted(user_dic[user].items(), key=lambda x: x[1], reverse=True)]
        if len(recommend) > 22:
            recommend = recommend[:22]
        output[user]=recommend
    rt=evaluate(output,name)
    print(rt)
    return -1*rt

# 過去からの推薦は最高でどれほどのスコアが出るのか
def analyse_past_recommend_score():
    score_dic={}
    for name in ['A','B','C','D']:
        predicts_idcg={}
        predicts_past={}
        with open('../data/personal/personal_test_items_IDCG_' + name + '.pickle', 'rb') as f:
            IDCG = pickle.load(f)
        with open('../data/personal/personal_train_' + name + '.pickle', 'rb') as f:
            train = pickle.load(f)
        for user in tqdm.tqdm(train.keys()):
            # 理想の推薦順を取得
            user_idcg=IDCG[user]
            del user_idcg['IDCG']
            ideal_recommend=[k for k, v in sorted(user_idcg.items(), key=lambda x: x[1], reverse=True)]

            # 過去の商品郡
            past_products=pd.unique(train[user]['product_id'])
            past_recommend=[]

            for i in ideal_recommend:
                if i in past_products:
                    past_recommend.append(i)

            if len(ideal_recommend)>22:
                ideal_recommend=ideal_recommend[:22]
            if len(past_recommend) > 22:
                past_recommend = past_recommend[:22]
            predicts_idcg[user]=ideal_recommend
            predicts_past[user]=past_recommend
        idcg_score=evaluate(predicts_idcg,name)
        past_score=evaluate(predicts_past, name)
        print(idcg_score)
        print(past_score)
        print(past_score/idcg_score)
        score_dic[name+'_idcg']=idcg_score
        score_dic[name + '_past'] = past_score
    print(score_dic)

#誤差関数
def DCG(user_id,items,personal_result):
    #IDCGが0の場合の分岐
    if personal_result[user_id]['IDCG']==0:
        return -1
    #まずDCGを計算
    DCG=0
    # 重複を除く
    new_items=[]
    for i in items:
        if i not in new_items:
            new_items.append(i)
    for i in range(len(new_items)):
        if new_items[i] in list(personal_result[user_id].keys()):
            DCG+=(2**personal_result[user_id][new_items[i]]-1)/np.log2(i+2)
    return DCG/personal_result[user_id]['IDCG']

#評価関数
def evaluate(predict,name):
    with open('../data/personal/personal_test_items_IDCG_' + name + '.pickle', 'rb') as f:
        personal_result = pickle.load(f)
    score=0.0
    count=0
    for i in predict.keys():
        tmp=DCG(i,predict[i],personal_result)
        if tmp==-1:
            count+=1
        else:
            score+=tmp
    return score/count

# dic for randomsearch
def dic_for_randomsaerch(name):
    train_data = read_data(name)
    # 時間減衰を読み込み
    with open('../data/time_weight/fitting_balanced_' + name + '.pickle', 'rb') as f:
        time_weight = pickle.load(f)
    test_min=datetime.datetime(year=2017, month=4, day=24)
    # 訓練期間のデータをフィルタ
    train_data = train_data[train_data['time_stamp'] <= datetime.datetime(year=2017, month=4, day=24)]
    personal_data = read_personal_train(name)

    unique_product_ids = list(pd.unique(train_data['product_id']))
    unique_user_ids = list(pd.unique(train_data['user_id']))

    # key:user, value:user's dic
    category_dic={}

    for user_id in tqdm.tqdm(unique_user_ids):
        user_dic={}
        for p_id in pd.unique(personal_data[user_id]['product_id']):
            # key:event, value:count
            product_dic={'conv':0,'click':0,'view':0,'cart':0}
            for _, row in personal_data[user_id][personal_data[user_id]['product_id'] == p_id].iterrows():
                if row['event_type'] == 1:
                    product_dic['view'] += 1 * time_weight[-1 * (row['time_stamp'] - test_min).days]
                elif row['event_type'] == 0:
                    product_dic['cart'] += 1 * time_weight[-1 * (row['time_stamp'] - test_min).days]
                elif row['event_type'] == 2:
                    product_dic['click'] += 1 * time_weight[-1 * (row['time_stamp'] - test_min).days]
                elif row['event_type'] == 3:
                    product_dic['conv'] += 1 * time_weight[-1 * (row['time_stamp'] - test_min).days]
            user_dic[p_id]=product_dic
        category_dic[user_id]=user_dic

    with open('../data/predict_myself/dic_for_randomsearch_' + name + '.pickle', 'wb') as f:
        pickle.dump(category_dic, f)

# ベイズ最適化用の関数
def category_weight_score(x):
    print(x)
    name='D'
    with open('../data/predict_myself/dic_for_randomsearch_' + name + '.pickle', 'rb') as f:
        category_dic=pickle.load(f)
    conv, click, view, cart = float(x[:, 0]), float(x[:, 1]),float(x[:, 2]), float(x[:, 3])
    predict_dic={}
    for user in category_dic.keys():
        score_dic={}
        for item in category_dic[user].keys():
            score=0
            score += category_dic[user][item]['conv'] * conv
            score += category_dic[user][item]['click'] * click
            score += category_dic[user][item]['view'] * view
            score += category_dic[user][item]['cart'] * cart
            score_dic[item]=score
        recommend = [k for k, v in sorted(score_dic.items(), key=lambda x: x[1], reverse=True)]
        if len(recommend)>22:
            recommend=recommend[:22]
        predict_dic[user]=recommend
    score=evaluate(predict_dic,name)
    print(score)
    return -1*score

# ベイズ最適化
def baysian_optimazation():
    bounds = [{'name': 'conv', 'type': 'continuous', 'domain': (0.0, 1.0)},
              {'name': 'click', 'type': 'continuous', 'domain': (0.0, 1.0)},
              {'name': 'view', 'type': 'continuous', 'domain': (0.0, 1.0)},
              {'name': 'cart', 'type': 'continuous', 'domain': (0.0, 1.0)},]
    # 事前探索を行います。
    opt_mnist = GPyOpt.methods.BayesianOptimization(f=category_weight_score, domain=bounds,initial_design_numdata=20,verbosity=True)

    # 最適なパラメータを探索します。
    opt_mnist.run_optimization(max_iter=100,verbosity=True)
    print("optimized parameters: {0}".format(opt_mnist.x_opt))
    print("optimized loss: {0}".format(opt_mnist.fx_opt))
# ベイズ最適化
def baysian_optimazation_for_fm():
    bounds = [{'name': 'l2_w', 'type': 'continuous', 'domain': (0.0, 1.0)},
              {'name': 'l2_v', 'type': 'continuous', 'domain': (0.0, 1.0)},
              {'name': 'l2', 'type': 'continuous', 'domain': (0.0, 1.0)},
              {'name': 'n_iter', 'type': 'discrete', 'domain': (100,200,300)},
              {'name': 'rank', 'type': 'discrete', 'domain': (2,4,8,16)},]
    # 事前探索を行います。
    opt_mnist = GPyOpt.methods.BayesianOptimization(f=fm_test, domain=bounds,initial_design_numdata=10,verbosity=True)

    # 最適なパラメータを探索します。
    opt_mnist.run_optimization(max_iter=30,verbosity=True)
    print("optimized parameters: {0}".format(opt_mnist.x_opt))
    print("optimized loss: {0}".format(opt_mnist.fx_opt))

if __name__=='__main__':
    #create_evaluate_matrix_optimize('C')
    extract_itemdata('B')