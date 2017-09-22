#coding:utf-8
#データの基本統計量を調べる

import numpy as np
import pandas as pd
from numba import jit
import time
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

# データ読み込み
def read_data(name):
    train = pd.read_csv('../data/train/train_'+name+'.tsv',delimiter='\t',parse_dates=['time_stamp'])
    return train
def read_personal_data(name):
    with open('../data/personal/personal_'+name+'.pickle', 'rb') as f:
        data=pickle.load(f)
    return data
def read_personal_test(name):
    with open('../data/personal/personal_test_'+name+'.pickle', 'rb') as f:
        data=pickle.load(f)
    return data
def read_personal_train(name):
    with open('../data/personal/personal_train_'+name+'.pickle', 'rb') as f:
        data=pickle.load(f)
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

# 訓練期間の評価値行列を作成(重み付け)
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

# -----時間と推薦商品の関係の可視化-----
def extract_time_and_past_items():
    for name in ['A','B','C','D']:
        train=read_personal_train(name)
        test=read_personal_test(name)
        idcg=read_personal_test_idcg(name)

        # トレイン期間とテスト期間の商品の時間の差分
        days=[]

        # 各ユーザに対して
        for user in tqdm.tqdm(idcg.keys()):
            train_unique_items=pd.unique(train[user]['product_id'])
            test_unique_items=pd.unique(test[user]['product_id'])
            # どちらにも存在するitemを抽出
            #item_subset=set(train_unique_items) & set(test_unique_items)

            # train期間におけるイベントの期間分布を調べる
            item_subset=train_unique_items

            for item in item_subset:
                # テスト期間の一番早い時間
                #test_min = test[user][test[user]['product_id']==item]['time_stamp'].min()
                # テスト開始期間に合わせる
                test_min=datetime.datetime(year=2017,month=4,day=24)

                # トレイン期間の全イベントのndarrayを作成
                train_ndarray=pd.to_datetime(train[user][train[user]['product_id']==item]['time_stamp'])-test_min
                for i in train_ndarray:
                    days.append(i.days)

        with open('../data/view/time_all_train_teststart_'+str(name)+'.pickle','wb') as f:
            pickle.dump(days,f)
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
            ('poly', PolynomialFeatures(degree=3)),
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
        with open('../data/view/time_all_train_teststart_' + str(name) + '.pickle', 'rb') as f:
            data=pickle.load(f)
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
        plt.savefig('../data/view/time_all_train_' + str(name) + '.png')
        plt.show()


if __name__=='__main__':
    #view_time()
    #extract_time_and_past_items()
    create_evaluate_matrix_time_weighted('D')