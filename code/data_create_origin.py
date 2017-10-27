#　データ作成を行うクラス
#coding:utf-8

import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm
import copy
import datetime
from pprint import pprint
from multiprocessing import Process,Manager
from sklearn.model_selection import KFold
from imblearn.ensemble import BalancedBaggingClassifier
from pprint import pprint

# 任意のブロック数に分割
def chunks(l, n):
    split_num=int(len(l)/n)+1
    return [l[i:i+split_num] for i in range(0, len(l), split_num)]

def datacreate_multi(name):
    train = pd.read_pickle('../data/personal/personal_train_' + name + '.pickle')
    keys=list(train.keys())
    keys=chunks(keys,8)
    manager = Manager()
    rt_data = manager.list()
    jobs = []
    for i in range(8):
        p = Process(target=conversion_data, args=(name, keys[i], rt_data))
        jobs.append(p)
    [x.start() for x in jobs]
    [x.join() for x in jobs]

    t_data=list(rt_data)
    output={}
    for i in t_data:
        output.update(i)
    with open('../data/conv_pred/train_X_'+name+'.pickle','wb') as f:
        pickle.dump(output,f)

# コンバージョンされるかどうかのデータ
def conversion_data(name,keys,rt_data):
    train = pd.read_pickle('../data/personal/personal_train_' + name + '.pickle')
    with open('../data/time_weight/fitting_balanced_'+name+'.pickle','rb') as f:
        time_weight = pickle.load(f)

    train_users={}
    dic_default={'score_conv':0,            # weight conv
                 'score_click': 0,          # weight click
                 'score_view': 0,           # weight view
                 'score_cart': 0,           # weight cart
                 'is_other_item_check': 0,  #
                 'day_item_per': 0,         #
                 'day_event_per': 0,        #
                 'continue_event': 0,       #
                 'is_last_event': 0,        #
                 'unique_item_num': 0,      #
                 'event_num': 0,            #
                 'conv_num': 0,             #
                 'time_length': 0,          #
                 'last_event_type': 0,      #
                 'is_conved': 0,            #
                 }
    test_min = datetime.datetime(year=2017, month=4, day=24)

    for user in tqdm(keys):
        train_items={}
        user_data=train[user]
        user_data=user_data.sort_values('time_stamp')
        final_event=user_data.max()[4]
        conv_num=len(user_data[user_data['event_type']==3])
        event_num=len(user_data)
        item_num=len(pd.unique(user_data['product_id']))
        continue_dic={}

        now_p=None
        now_count=0
        for i_product in user_data['product_id']:
            if i_product not in continue_dic.keys():
                continue_dic[i_product]=0
            if now_p != i_product:
                if now_p != None:
                    continue_dic[now_p]=now_count
                now_p=i_product
                now_count=1
            else:
                now_count+=1

        for item in pd.unique(user_data['product_id']):
            item_data=user_data[user_data['product_id']==item]
            item_dic=copy.deepcopy(dic_default)
            item_final=item_data.max()[4]
            item_dic['time_length']=(test_min-item_final).total_seconds()/3600
            for _,row in item_data.iterrows():
                if row['event_type'] == 1:
                    item_dic['score_view'] += time_weight[(test_min - row['time_stamp']).days]
                elif row['event_type'] == 0:
                    item_dic['score_cart'] += time_weight[(test_min - row['time_stamp']).days]
                elif row['event_type'] == 2:
                    item_dic['score_click'] += time_weight[(test_min - row['time_stamp']).days]
                elif row['event_type'] == 3:
                    item_dic['score_conv'] += time_weight[(test_min - row['time_stamp']).days]
                    item_dic['is_conved']=1
            item_dic['last_event_type']=str(row['event_type'])
            if row['time_stamp']== final_event:
                item_dic['is_last_event']=1
            item_dic['continue_event']=continue_dic[item]
            item_dic['unique_item_num']=item_num
            item_dic['conv_num'] = conv_num
            item_dic['event_num'] = event_num

            train_items[item]=item_dic

        train_users[user]=train_items
    rt_data.append(train_users)

def datacreate_test_multi(name):
    train=pd.read_pickle('../data/personal/personal_'+name+'.pickle')
    keys=list(train.keys())
    keys=chunks(keys,8)
    manager = Manager()
    rt_data = manager.list()
    jobs = []
    for i in range(8):
        p = Process(target=conversion_test_data, args=(name, keys[i], rt_data))
        jobs.append(p)
    [x.start() for x in jobs]
    [x.join() for x in jobs]

    t_data=list(rt_data)
    output={}
    for i in t_data:
        output.update(i)
    with open('../data/conv_pred/test_X_cut_origin_'+name+'.pickle','wb') as f:
        pickle.dump(output,f)

# コンバージョンされるかどうかのデータ
def conversion_test_data(name,keys,rt_data):
    train = pd.read_pickle('../data/personal/personal_' + name + '.pickle')
    with open('../data/time_weight/fitting_balanced_'+name+'.pickle','rb') as f:
        time_weight = pickle.load(f)

    train_users={}
    dic_default={'score_conv':0,
                 'score_click': 0,
                 'score_view': 0,
                 'score_cart': 0,
                 'is_other_item_check': 0,
                 'day_item_per': 0,
                 'day_event_per': 0,
                 'continue_event': 0,
                 'is_last_event': 0,
                 'unique_item_num': 0,
                 'event_num': 0,
                 'conv_num': 0,
                 'time_length': 0,
                 'last_event_type': 0,
                 'is_conved': 0,
                 }
    test_min = datetime.datetime(year=2017, month=5, day=1)

    for user in tqdm(keys):
        train_items={}
        user_data=train[user]
        user_data=user_data.sort_values('time_stamp')
        user_data=user_data[user_data['time_stamp']>datetime.datetime(year=2017, month=4, day=7)]
        final_event=user_data.max()[4]
        conv_num=len(user_data[user_data['event_type']==3])
        event_num=len(user_data)
        item_num=len(pd.unique(user_data['product_id']))
        continue_dic={}

        now_p=None
        now_count=0
        for i_product in user_data['product_id']:
            if i_product not in continue_dic.keys():
                continue_dic[i_product]=0
            if now_p != i_product:
                if now_p != None:
                    continue_dic[now_p]=now_count
                now_p=i_product
                now_count=1
            else:
                now_count+=1

        for item in pd.unique(user_data['product_id']):
            item_data=user_data[user_data['product_id']==item]
            item_dic=copy.deepcopy(dic_default)
            item_final=item_data.max()[4]
            item_dic['time_length']=(test_min-item_final).total_seconds()/3600
            for _,row in item_data.iterrows():
                if row['event_type'] == 1:
                    item_dic['score_view'] += time_weight[(test_min - row['time_stamp']).days]
                elif row['event_type'] == 0:
                    item_dic['score_cart'] += time_weight[(test_min - row['time_stamp']).days]
                elif row['event_type'] == 2:
                    item_dic['score_click'] += time_weight[(test_min - row['time_stamp']).days]
                elif row['event_type'] == 3:
                    item_dic['score_conv'] += time_weight[(test_min - row['time_stamp']).days]
                    item_dic['is_conved']=1
            item_dic['last_event_type']=str(row['event_type'])
            if row['time_stamp']== final_event:
                item_dic['is_last_event']=1
            item_dic['continue_event']=continue_dic[item]
            item_dic['unique_item_num']=item_num
            item_dic['conv_num'] = conv_num
            item_dic['event_num'] = event_num

            train_items[item]=item_dic

        train_users[user]=train_items
    rt_data.append(train_users)


# 予測値を作成する関数
def conversion_y(name):
    test = pd.read_pickle('../data/personal/personal_test_' + name + '.pickle')
    with open('../data/conv_pred/train_X_'+name+'.pickle', 'rb') as f:
        created_data=pickle.load(f)
    train_X=[]
    train_y=[]
    for user in tqdm(created_data.keys()):
        if len(created_data[user])==0:
            continue
        for item in created_data[user].keys():
            train_X.append(created_data[user][item])
            tmp_dic=test[user][test[user]['product_id']==item]
            if  len(pd.unique(tmp_dic['event_type']))==0:
                train_y.append(0)
            elif max(pd.unique(tmp_dic['event_type']))==3:
                train_y.append(1)
            else:
                train_y.append(0)
    output={'X':train_X,'y':train_y}
    with open('../data/conv_pred/train_data_'+name+'.pickle','wb') as f:
        pickle.dump(output,f)

def randomforest(name):
    with open('../data/conv_pred/train_data_'+'A'+'.pickle','rb') as f:
        data=pickle.load(f)
    model=RandomForestClassifier(n_estimators=500,max_features=0.6,verbose=1,n_jobs=8,oob_score=True)
    v=DictVectorizer()
    X=v.fit_transform(data['X'])
    model.fit(X,data['y'])
    print(model.oob_score_)

def cross_validation(name):
    with open('../data/conv_pred/train_data_'+name+'.pickle','rb') as f:
        data=pickle.load(f)
    v=DictVectorizer()
    X=v.fit_transform(data['X'])
    y=np.array(data['y'])
    kf=KFold(n_splits=10)
    fscore=0
    ftscore=0
    for train_index,test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #model = RandomForestClassifier(n_estimators=100, n_jobs=8,class_weight={0:1,1:3000})
        model = BalancedBaggingClassifier(n_estimators=500,n_jobs=8,verbose=1)
        model.fit(X_train,y_train)
        predict=model.predict_proba(X_test)
        score,t_score=eval(y_test,predict)
        #pprint(sorted(zip(np.mean([est.steps[1][1].feature_importances_ for est in model.estimators_], axis=0),v.feature_names_),key=lambda x:x[0],reverse=True))
        print('score : ',str(score))
        print('true_score : ', str(t_score))
        fscore+=score
        ftscore+=t_score
    print('\n')
    print('final score : ',str(fscore/10))
    print('final true_score : ', str(ftscore / 10))

def eval(test,pred):
    score=0
    s_count=0
    true_true=0
    t_count=0
    for i in range(len(test)):
        if pred[i][1]>=0.50:
            s_count+=1
            if test[i]==1:
                score+=1
        if test[i]==1:
            t_count+=1
            if pred[i][1]>=0.50:
                true_true+=1
    return score/s_count, true_true/t_count

datacreate_test_multi('A')
datacreate_test_multi('B')