# 推薦コンペ用

## フォルダ別の役割
* analysis - 分析用コード
* code - 予測用コード
* data - 各種データ

## 方針
* とりあえず各人の特徴量を多数作成する
* 24日学習，6日テストの交差検定を行い誤差を測定
* rating情報があればALSやBPRを用いた推薦が出来る
* rating情報がなければ，rating情報を行動データから予測するしかない？
* 直近の行動情報は次の期間への影響が大きいのではないか？

## 仮説
* カートに入れた商品は購入されやすい
* 短期間に沢山閲覧する商品は買われやすい
* 似たジャンルの商品は買われやすい

## 確認すること
* 似たジャンルの商品は購入されるのか
* 購入，クリック，閲覧はすべて同じ商品の傾向はあるのかどうか?
* 1回買った商品はその後購入，クリック，閲覧されるのか？

## 優先順位
1. 交差検定環境の構築
2. 情報可視化
3. 誤差の縮小
4. 新たな仮説とモデルの構築＆アンサンブル

## メモ
予測する人数の合計: 11598
- Aの人数: 7264
- Bの人数: 2366
- Cの人数: 1648
- Dの人数: 320

## 実行時の手順
1. analysis/basic_statistic_analysis.pyのextractallを実行