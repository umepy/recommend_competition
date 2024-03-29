#　データ作成を行うクラス
#coding:utf-8

import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
import copy
import datetime
from pprint import pprint
from multiprocessing import Process,Manager
from sklearn.model_selection import KFold
from imblearn.ensemble import BalancedBaggingClassifier,BalanceCascade
from pprint import pprint
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import GPyOpt
from sklearn.tree import DecisionTreeClassifier

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
    with open('../data/personal/item_' + name + '.pickle', 'rb') as f:
        item_dic_data=pickle.load(f)

    train_users={}
    dic_default={'score_conv':0,            # weight conv
                 'score_click': 0,          # weight click
                 'score_view': 0,           # weight view
                 'score_cart': 0,           # weight cart
                 'day_item_per': 0,         # None - 同じ日の商品割合
                 'day_event_per': 0,        # None - 同じ日のイベント割合
                 'continue_event': 0,       # 連続イベント数
                 'is_last_event': 0,        # その人の最終イベントか？
                 'unique_item_num': 0,      # ユニークアイテム数
                 'event_num': 0,            # イベント数
                 'conv_num': 0,             # 購入数
                 'time_length': 0,          # 最終イベントがどれくらい離れているか
                 'last_event_type': 0,      # 最終イベントタイプ
                 'is_conved': 0,            # 今までで購入されているか
                 'percentage_conv':0,       # 購入数 / イベント数
                 'percentage_uni_item': 0,  # ユニークアイテム数 / イベント数
                 # 'item_evented_by_users':0, # その商品に行動を起こしたユーザ数
                 # 'item_event_each_users': 0,# その商品の1ユーザーあたりの行動数
                 # 'item_conv_num_by_users':0,# その商品の購入数
                 # 'item_conved_by_event': 0, # その商品の購入されている割合 (conv / event)
                 # 'item_conved_by_users': 0, # その商品の購入されている割合 (conv / user)
                 # 'hot_item_by_users':0,     # 流行の購入される商品なのか？ (conv * time_weight)
                 # 'hot_item_each_users': 0,  # 流行の購入される商品なのか？ (conv * time_weight) / user_num
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
            item_dic['percentage_conv']=conv_num/float(event_num)
            item_dic['percentage_unique_item'] = item_num / float(event_num)

            # item 固有の情報
            # tmp_item_dic=item_dic_data[item]
            # tmp_item_dic=tmp_item_dic[tmp_item_dic['time_stamp']<test_min]
            # user_num=len(pd.unique(tmp_item_dic['user_id']))
            # event_num=len(tmp_item_dic)
            # conv_num=len(tmp_item_dic[tmp_item_dic['event_type']==3])
            # item_dic['item_evented_by_users']= user_num
            # item_dic['item_event_each_users'] = event_num/user_num
            # item_dic['item_conv_num_by_users'] = conv_num
            # item_dic['item_conved_by_event'] = conv_num/event_num
            # item_dic['item_conved_by_users'] = conv_num/user_num
            #
            # tmp_item_dic=tmp_item_dic[tmp_item_dic['event_type']==3]
            # for _,row in tmp_item_dic.iterrows():
            #     item_dic['hot_item_by_users'] += time_weight[(test_min - row['time_stamp']).days]
            # item_dic['hot_item_each_users']=item_dic['hot_item_by_users']/user_num

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
    with open('../data/conv_pred/test_X_cut_'+name+'.pickle','wb') as f:
        pickle.dump(output,f)

# コンバージョンされるかどうかのデータ
def conversion_test_data(name,keys,rt_data):
    train = pd.read_pickle('../data/personal/personal_' + name + '.pickle')
    with open('../data/time_weight/fitting_balanced_'+name+'.pickle','rb') as f:
        time_weight = pickle.load(f)
    with open('../data/personal/item_' + name + '.pickle', 'rb') as f:
        item_dic_data=pickle.load(f)

    train_users={}
    dic_default = {'score_conv': 0,  # weight conv
                   'score_click': 0,  # weight click
                   'score_view': 0,  # weight view
                   'score_cart': 0,  # weight cart
                   'day_item_per': 0,  # None - 同じ日の商品割合
                   'day_event_per': 0,  # None - 同じ日のイベント割合
                   'continue_event': 0,  # 連続イベント数
                   'is_last_event': 0,  # その人の最終イベントか？
                   'unique_item_num': 0,  # ユニークアイテム数
                   'event_num': 0,  # イベント数
                   'conv_num': 0,  # 購入数
                   'time_length': 0,  # 最終イベントがどれくらい離れているか
                   'last_event_type': 0,  # 最終イベントタイプ
                   'is_conved': 0,  # 今までで購入されているか
                   'percentage_conv': 0,  # 購入数 / イベント数
                   'percentage_uni_item': 0,  # ユニークアイテム数 / イベント数
                   'item_evented_by_users': 0,  # その商品に行動を起こしたユーザ数
                   'item_event_each_users': 0,  # その商品の1ユーザーあたりの行動数
                   'item_conv_num_by_users': 0,  # その商品の購入数
                   'item_conved_by_event': 0,  # その商品の購入されている割合 (conv / event)
                   'item_conved_by_users': 0,  # その商品の購入されている割合 (conv / user)
                   'hot_item_by_users': 0,  # 流行の購入される商品なのか？ (conv * time_weight)
                   'hot_item_each_users': 0,  # 流行の購入される商品なのか？ (conv * time_weight) / user_num
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

            item_dic['percentage_conv'] = conv_num / float(event_num)
            item_dic['percentage_unique_item'] = item_num / float(event_num)

            # # item 固有の情報
            # tmp_item_dic = item_dic_data[item]
            # tmp_item_dic = tmp_item_dic[tmp_item_dic['time_stamp']>datetime.datetime(year=2017, month=4, day=7)]
            # user_num = len(pd.unique(tmp_item_dic['user_id']))
            # event_num = len(tmp_item_dic)
            # conv_num = len(tmp_item_dic[tmp_item_dic['event_type'] == 3])
            # item_dic['item_evented_by_users'] = user_num
            # item_dic['item_event_each_users'] = event_num / user_num
            # item_dic['item_conv_num_by_users'] = conv_num
            # item_dic['item_conved_by_event'] = conv_num / event_num
            # item_dic['item_conved_by_users'] = conv_num / user_num
            #
            # tmp_item_dic = tmp_item_dic[tmp_item_dic['event_type'] == 3]
            # for _, row in tmp_item_dic.iterrows():
            #     item_dic['hot_item_by_users'] += time_weight[(test_min - row['time_stamp']).days]
            # item_dic['hot_item_each_users'] = item_dic['hot_item_by_users'] / user_num

            train_items[item]=item_dic

        train_users[user]=train_items
    rt_data.append(train_users)


# 予測値を作成する関数
def conversion_y(name):
    test = pd.read_pickle('../data/personal/personal_train_' + name + '.pickle')
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
            #tmp_dic=tmp_dic[tmp_dic['time_stamp']>datetime.datetime(year=2017, month=4, day=24)]
            if  len(pd.unique(tmp_dic['event_type']))==0:
                train_y.append(1)
            # elif max(pd.unique(tmp_dic['event_type']))==3:
            #     train_y.append(1)
            else:
                train_y.append(0)
    output={'X':train_X,'y':train_y}
    with open('../data/conv_pred/super_train_notRec_'+name+'.pickle','wb') as f:
        pickle.dump(output,f)

# 予測値を作成する関数
def conversion_positive_y(name):
    test = pd.read_pickle('../data/personal/personal_test_' + name + '.pickle')
    with open('../data/conv_pred/train_X_'+name+'.pickle', 'rb') as f:
        created_data=pickle.load(f)
    train_X=[]
    train_y=[]
    for user in tqdm(created_data.keys()):
        if len(created_data[user])==0:
            continue
        for item in created_data[user].keys():
            tmp_dic=test[user][test[user]['product_id']==item]
            if  len(pd.unique(tmp_dic['event_type']))==0:
                continue
            elif max(pd.unique(tmp_dic['event_type']))==3:
                train_X.append(created_data[user][item])
                train_y.append(8)
            elif max(pd.unique(tmp_dic['event_type'])) == 2:
                train_X.append(created_data[user][item])
                train_y.append(4)
            else:
                train_X.append(created_data[user][item])
                train_y.append(0)
    output={'X':train_X,'y':train_y}
    with open('../data/conv_pred/train_reg_'+name+'.pickle','wb') as f:
        pickle.dump(output,f)

def randomforest(name):
    with open('../data/conv_pred/train_data_'+'A'+'.pickle','rb') as f:
        data=pickle.load(f)
    model=RandomForestClassifier(n_estimators=500,max_features=0.6,verbose=1,n_jobs=8,oob_score=True)
    v=DictVectorizer()
    X=v.fit_transform(data['X'])
    model.fit(X,data['y'])
    print(model.oob_score_)

def cross_validation(x):
    with open('../data/conv_pred/train_data_'+x+'.pickle','rb') as f:
        data=pickle.load(f)
    print(data)
    v=DictVectorizer()
    X=v.fit_transform(data['X'])
    y=np.array(data['y'])

    zero=0
    one=0
    for i in y:
        if i==0:
            zero+=1
        else:
            one+=1
    print(zero)
    print(one)

    cv=5
    kf=KFold(n_splits=cv)
    fscore=0
    ftscore=0
    all_f_value=0
    all_prec=0
    for train_index,test_index in tqdm(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #model = RandomForestRe(n_estimators=100, n_jobs=8)
        model = BalancedBaggingClassifier(n_estimators=100,n_jobs=8)
        #model = xgb.XGBClassifier(n_estimators=500,max_delta_step=1,scale_pos_weight=zero/one)
        model.fit(X_train,y_train)
        predict=model.predict_proba(X_test)
        precision,recall,f_value,all_pre=eval(y_test,predict)
        all_prec+=all_pre
        fscore+=precision
        ftscore+=recall
        all_f_value+=f_value
    pprint(sorted(
        zip(np.mean([est.steps[1][1].feature_importances_ for est in model.estimators_], axis=0), v.feature_names_),
        key=lambda x: x[0], reverse=True))
    print('\n')
    print('final precision : ',str(fscore/cv))
    print('final recall : ', str(ftscore / cv))
    print('final f-value : ', str(all_f_value / cv))
    print('final all_precision : ', str(all_prec / cv))

def cross_validation_ensenble(name):
    with open('../data/conv_pred/train_data2_'+name+'.pickle','rb') as f:
        data=pickle.load(f)
    v=DictVectorizer()
    X=v.fit_transform(data['X'])
    y=np.array(data['y'])

    cv=5
    kf=KFold(n_splits=cv)
    fscore=0
    ftscore=0
    all_f_value=0
    for train_index,test_index in tqdm(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ensenble_num=1

        bc = BalanceCascade(estimator=LogisticRegression(),n_max_subset=ensenble_num)
        bc_x, bc_y = bc.fit_sample(X_train, y_train)
        models=[]
        predicts=[]
        final_result=[]
        models.append(xgb.XGBClassifier(n_estimators=500,max_delta_step=1))
        for i in range(ensenble_num):
            models[i].fit(bc_x[i],bc_y[i])
        for i in range(ensenble_num):
            predicts.append(models[i].predict_proba(X_test))
        for i in range(len(predicts[0])):
            result=[0,0]
            for j in range(ensenble_num):
                result[0]+=predicts[j][i][0]/ensenble_num
                result[1] += predicts[j][i][1]/ensenble_num
            final_result.append(result)
        precision,recall,f_value,_=eval(y_test,final_result)
        fscore+=precision
        ftscore+=recall
        all_f_value+=f_value
    # pprint(sorted(
    #     zip(np.mean([est.steps[1][1].feature_importances_ for est in model.estimators_], axis=0), v.feature_names_),
    #     key=lambda x: x[0], reverse=True))
    print('\n')
    print('final precision : ',str(fscore/cv))
    print('final recall : ', str(ftscore / cv))
    print('final f-value : ', str(all_f_value / cv))

def eval(test,pred):
    assert len(test)==len(pred)
    precision=0
    p_count=0
    recall=0
    r_count=0
    all_prec=0
    for i in range(len(test)):
        if pred[i][1]>0.50:
            p_count+=1
            if test[i]==1:
                precision+=1
        if test[i]==1:
            r_count+=1
            if pred[i][1]>=0.50:
                recall+=1
        if test[i]==0 and pred[i][0]>=0.5:
            all_prec+=1
        if test[i]==1 and pred[i][1]>0.5:
            all_prec+=1
    precision/=p_count
    recall/=r_count
    return precision, recall, 2*precision*recall/(precision+recall),all_prec/len(test)

def baysian_optimazation():
    bounds = [{'name': 'min_chiled', 'type': 'continuous', 'domain': (0.0, 1.0)},
              {'name': 'l2_v', 'type': 'continuous', 'domain': (0.0, 1.0)},
              {'name': 'l2', 'type': 'continuous', 'domain': (0.0, 1.0)},
              {'name': 'n_iter', 'type': 'discrete', 'domain': (100,200,300)},
              {'name': 'rank', 'type': 'discrete', 'domain': (2,4,8,16)},]
    # 事前探索を行います。
    opt_mnist = GPyOpt.methods.BayesianOptimization(f=cross_validation, domain=bounds,initial_design_numdata=10,verbosity=True)

    # 最適なパラメータを探索します。
    opt_mnist.run_optimization(max_iter=30,verbosity=True)
    print("optimized parameters: {0}".format(opt_mnist.x_opt))
    print("optimized loss: {0}".format(opt_mnist.fx_opt))

def cross_validation_another(x):
    with open('../data/conv_pred/super_train_data_day_'+'A'+'.pickle','rb') as f:
        data=pickle.load(f)
    with open('../data/conv_pred/super_test_data_day_'+'A'+'.pickle','rb') as f:
        test=pickle.load(f)
    v=DictVectorizer()
    X_train=v.fit_transform(data['X'])
    y_train=np.array(data['y'])
    X_test = v.transform(test['X'])
    y_test = np.array(test['y'])
    zero=0
    one=0
    for i in y_train:
        if i==0:
            zero+=1
        else:
            one+=1
    print(zero)
    print(one)

    model = BalancedBaggingClassifier(n_estimators=100,n_jobs=8,max_samples=0.6)
    #model = xgb.XGBClassifier(n_estimators=500, max_delta_step=1, scale_pos_weight=zero / one)
    model.fit(X_train,y_train)
    predict=model.predict_proba(X_test)
    precision,recall,f_value,all_pre=eval(y_test,predict)
    all_prec=all_pre
    fscore=precision
    ftscore=recall
    all_f_value=f_value
    print('\n')
    print('final precision : ',str(fscore))
    print('final recall : ', str(ftscore))
    print('final f-value : ', str(all_f_value))
    print('final all_precision : ', str(all_prec ))

#conversion_positive_y('A')
cross_validation_another('A')