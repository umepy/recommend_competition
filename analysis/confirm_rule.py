#coding:utf-8
#sample submitを調査してルールの確認

import pandas as pd
import numpy as np

#予測者数の確認（全員予測するのか？）
df = pd.read_csv('../data/sample_submit.tsv',delimiter='\t',names=['user_id','item_id','rank'])
print('予測する人数の合計: '+str(len(df[df['rank']==0])))
a= df[df.apply(lambda x: 'A' in x['user_id'],axis=1)]
b= df[df.apply(lambda x: 'B' in x['user_id'],axis=1)]
c= df[df.apply(lambda x: 'C' in x['user_id'],axis=1)]
d= df[df.apply(lambda x: 'D' in x['user_id'],axis=1)]
print('Aの人数: '+str(len(a[a['rank']==0])))
print('Bの人数: '+str(len(b[b['rank']==0])))
print('Cの人数: '+str(len(c[c['rank']==0])))
print('Dの人数: '+str(len(d[d['rank']==0])))