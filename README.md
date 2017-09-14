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
* 直近の行動情報は次の期間への影響が大きいのではないか？→直近の行動から予測するモデル
* 数個のモデルのアンサンブルを用いる

## モデルアイデア
* 最近傍推薦-ユーザ間の類似度をどのように測るか、購買した商品の個数？

## 仮説
* カートに入れた商品は購入されやすい
* 短期間に沢山閲覧する商品は買われやすい
* 似たジャンルの商品は買われやすい
* 広告経由でのコンバージョン評価→潜在的に欲しい商品を推薦するべき
* 直近の行動データの商品は推薦すべき

## 問題点
* 過去の閲覧等を評価済みとして，それ以外を推薦してしまうのは良くない？
* アンサンブルで予測を出力する場合，順位付けはどうするのか？
* 過去の行動データでの商品数が22未満の場合，類似した個人客の購入商品を推薦する

## 確認すること
* 似たジャンルの商品は購入されるのか
* 購入，クリック，閲覧はすべて同じ商品の傾向はあるのかどうか?
* 1回買った商品はその後購入，クリック，閲覧されるのか？

## 優先順位
1. 交差検定環境の構築
2. 情報可視化
3. 誤差の縮小
4. 新たな仮説とモデルの構築＆アンサンブル

## リマインド
extract_rankingをA,C,Dに対して実行 - 完

## メモ
予測する人数の合計: 11598
- Aの人数: 7264
- Bの人数: 2366
- Cの人数: 1648
- Dの人数: 320

## 結果その1
ランダムチョイス選択(過去の行動に関連した商品からのみのランダム推薦, mean of 5x 5CrossValidation)
* A - 0.1198
* B - 0.1379
* C - 0.1455
* D - 0.0590
* 加重平均 - 0.1255
* コンペサイト評価 - 0.1526

## 結果その2
簡単な重み付け選択推薦(過去の行動に関連した商品からそのうち最大のイベントを重みとした推薦, mean of 5x 5CrossValidation)
* A - 0.1268
* B - 0.1589
* C - 0.1485
* D - 0.05765
* 加重平均 - 0.1345

## 結果その3
簡単な重み付け選択推薦(その2の手法のうち購入重みを無視した推薦, mean of 5x 5CrossValidation)
* A - 0.1190
* B - 0.1208
* C - 0.1478
* D - 0.05747
* 加重平均 - 0.1218

## 結果その4
簡単な重み付け選択推薦(カート商品のみを用いた順位付け, mean of 5x 5CrossValidation)
* A - 0.1388
* B - 0.1642
* C - 0.1448
* D - 0.06769
* 加重平均 - 0.1429

購入済みを推薦しない
* A - 0.1343
* B - 0.1262
* C - 0.1455
* D - 0.06786
* 加重平均 - 0.1324

## 結果その5
簡単な重み付け選択推薦(クリックのみを用いた順位付け, mean of 5x 5CrossValidation)
* A - 0.1225
* B - 0.1404
* C - 0.1482
* D - 0.05804
* 加重平均 - 0.1280

購入済みを推薦しない
* A - 0.1185
* B - 0.1207
* C - 0.1480
* D - 0.05843
* 加重平均 - 0.1215

## 結果その6
簡単な重み付け選択推薦(閲覧のみを用いた順位付け, mean of 5x 5CrossValidation)
* A - 0.1584
* B - 0.1817
* C - 0.1836
* D - 0.06116
* 加重平均 - 0.1641
* コンペ結果 - 0.2056

購入済みを推薦しない
* A - 0.1530
* B - 0.1505
* C - 0.1836
* D - 0.06119
* 加重平均 - 0.1543

## 実行時の手順
1. analysis/basic_statistic_analysis.pyのextract_allを実行(personalデータの作成)