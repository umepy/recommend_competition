import pandas as pd
import numpy as np
import pickle
from multiprocessing import Process,Manager
import datetime
from tqdm import tqdm
import copy
from sklearn.feature_extraction import DictVectorizer
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import KFold
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
    #return output
    with open('../data/conv_pred/train_X3_'+name+'.pickle','wb') as f:
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
                 'continue_event': 0,       # 連続イベント数
                 'is_last_event': 0,        # その人の最終イベントか？
                 'unique_item_num': 0,      # ユニークアイテム数
                 'event_num': 0,            # イベント数
                 'conv_num': 0,             # 購入数
                 'time_length': 0,          # 最終イベントがどれくらい離れているか
                 'last_event_type': 0,      # 最終イベントタイプ
                 'is_conved': 0,            # 今までで購入されているか
                 'item_start':0,
                 'item_end':0
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
            tmp_item_dic=item_dic_data[item]
            tmp_item_dic=tmp_item_dic[tmp_item_dic['time_stamp']<test_min]
            item_dic['item_start'] = (test_min - tmp_item_dic.min()[4]).total_seconds()/3600
            item_dic['item_end'] = (test_min - tmp_item_dic.max()[4]).total_seconds()/3600


            train_items[item]=item_dic

        train_users[user]=train_items
    rt_data.append(train_users)

def conversion_y(name):
    test = pd.read_pickle('../data/personal/personal_test_' + name + '.pickle')
    created_data = datacreate_multi(name)
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
    return output

def datacreate_test_multi(name):
    train = pd.read_pickle('../data/personal/personal_' + name + '.pickle')
    keys=list(train.keys())
    keys=chunks(keys,8)
    manager = Manager()
    rt_data = manager.list()
    jobs = []
    for i in range(8):
        p = Process(target=conversion_data_test, args=(name, keys[i], rt_data))
        jobs.append(p)
    [x.start() for x in jobs]
    [x.join() for x in jobs]

    t_data=list(rt_data)
    output={}
    for i in t_data:
        output.update(i)
    #return output
    with open('../data/conv_pred/test_X3_'+name+'.pickle','wb') as f:
        pickle.dump(output,f)

# コンバージョンされるかどうかのデータ
def conversion_data_test(name,keys,rt_data):
    train = pd.read_pickle('../data/personal/personal_' + name + '.pickle')
    with open('../data/time_weight/fitting_balanced_'+name+'.pickle','rb') as f:
        time_weight = pickle.load(f)
    with open('../data/personal/item_' + name + '.pickle', 'rb') as f:
        item_dic_data=pickle.load(f)

    train_users={}
    dic_default={'score_conv':0,            # weight conv
                 'score_click': 0,          # weight click
                 'score_view': 0,           # weight view
                 'score_cart': 0,           # weight cart
                 'continue_event': 0,       # 連続イベント数
                 'is_last_event': 0,        # その人の最終イベントか？
                 'unique_item_num': 0,      # ユニークアイテム数
                 'event_num': 0,            # イベント数
                 'conv_num': 0,             # 購入数
                 'time_length': 0,          # 最終イベントがどれくらい離れているか
                 'last_event_type': 0,      # 最終イベントタイプ
                 'is_conved': 0,            # 今までで購入されているか
                 'item_start':0,
                 'item_end':0
                 }
    test_min = datetime.datetime(year=2017, month=5, day=1)

    for user in tqdm(keys):
        train_items={}
        user_data=train[user]
        user_data=user_data.sort_values('time_stamp')
        user_data = user_data[user_data['time_stamp'] > datetime.datetime(year=2017, month=4, day=7)]
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
            tmp_item_dic=item_dic_data[item]
            tmp_item_dic=tmp_item_dic[tmp_item_dic['time_stamp']>datetime.datetime(year=2017, month=4, day=7)]
            item_dic['item_start'] = (test_min - tmp_item_dic.min()[4]).total_seconds()/3600
            item_dic['item_end'] = (test_min - tmp_item_dic.max()[4]).total_seconds()/3600


            train_items[item]=item_dic

        train_users[user]=train_items
    rt_data.append(train_users)

def cross_validation(name):
    data=conversion_y(name)
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

datacreate_test_multi('A')
datacreate_test_multi('B')