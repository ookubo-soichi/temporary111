import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

trn = pd.read_csv('../input/train_ver2.csv')

for col in trn.columns:
    print('{}\n'.format(trn[col].head()))

# 6. カテゴリ変数の確認（コード2-7）
num_cols = [col for col in trn.columns[:24] if trn[col].dtype in ['int64', 'float64']]
trn[num_cols].describe()

# 7. カテゴリ変数の固有値を表示
cat_cols = [col for col in trn.columns[:24] if trn[col].dtype in ['O']]
trn[cat_cols].describe()
for col in cat_cols:
    uniq = np.unique(trn[col].astype(str))
    print('-' * 50)
    print('# col {}, n_uniq {}, uniq {}'.format(col, len(uniq), uniq))

# 8. 変数を棒グラフで可視化（コード2-9）
skip_cols = ['ncodpers', 'renta']
for col in trn.columns:
    # 出力に時間がかかりすぎる2つの変数をskipします。
    if col in skip_cols:
        continue
    # 見やすくするため、領域区分と変数名を出力します。
    print('='*50)
    print('col : ', col)
    # グラフの大きさ(figsize)を設定します。
    f, ax = plt.subplots(figsize=(20, 15))
    # seabornを使用した棒グラフを設定します。
    sns.countplot(x=col, data=trn, alpha=0.5)
    plt.show()

col = 'ind_recibo_ult1'
f, ax = plt.subplots(figsize=(20, 15))
sns.countplot(x=col, data=trn, alpha=0.5)
plt.show()
    
# 9. 月別金融商品の保有データを累積棒グラフで可視化します。（コード2-10）
# 日付データを基準として分析するため、日付データを別途に抽出します。
months = np.unique(trn['fecha_dato']).tolist()
# 商品変数24個を抽出します。
label_cols = trn.columns[24:].tolist()
label_over_time = []
for i in range(len(label_cols)):
    # 毎月、各商品の合計をlabel_sumに保存します。
    label_sum = trn.groupby(['fecha_dato'])[label_cols[i]].agg('sum')
    label_over_time.append(label_sum.tolist())
label_sum_over_time = []
for i in range(len(label_cols)):
    # 累積棒グラフを可視化するため、累積値を計算します。
    label_sum_over_time.append(np.asarray(label_over_time[i:]).sum(axis=0))
# 可視化のため色を指定します。
color_list = ['#F5B7B1','#D2B4DE','#AED6F1','#A2D9CE','#ABEBC6','#F9E79F','#F5CBA7','#CCD1D1']
f, ax = plt.subplots(figsize=(30, 15))
for i in range(len(label_cols)):
    # 24個の商品についてヒストグラムを描きます。
    sns.barplot(x=months, y=label_sum_over_time[i], color = color_list[i%8], alpha=0.7)
# 右上にLegendを追加します。
plt.legend([plt.Rectangle((0,0),1,1,fc=color_list[i%8], edgecolor = 'none') for i in range(len(label_cols))], label_cols, loc=1, ncol = 2, prop={'size':16})
plt.show()

# 10. 相対値を用いた積層棒グラフの可視化（コード2－11）
# label_sum_over_timeの値をパーセントに変換します。
label_sum_percent = (label_sum_over_time / (1.*np.asarray(label_sum_over_time).max(axis=0))) * 100
# 前のコードと同一の、可視化実行コードです。
f, ax = plt.subplots(figsize=(30, 15))
for i in range(len(label_cols)):
    sns.barplot(x=months, y=label_sum_percent[i], color = color_list[i%8], alpha=0.7)
plt.legend([plt.Rectangle((0,0),1,1,fc=color_list[i%8], edgecolor = 'none') for i in range(len(label_cols))], label_cols, loc=1, ncol = 2, prop={'size':16})

# 11. 商品変数をprodsにlist形式で保存します。（コード2－12）
prods = trn.columns[24:].tolist()
# 日付を数字に変換する変数です。2015-01-28は1、2016-06-28は18に変換されます。
def date_to_int(str_date):
    Y, M, D = [int(a) for a in str_date.strip().split("-")]
    int_date = (int(Y) - 2015) * 12 + int(M)
    return int_date
# 日付を数字に変換し、int_dateに保存します。
trn['int_date'] = trn['fecha_dato'].map(date_to_int).astype(np.int8)
# データをコピーし、int_dateの日付に1を加えてlagを生成します。変数名に_prevを追加します。
trn_lag = trn.copy()
trn_lag['int_date'] += 1
trn_lag.columns = [col + '_prev' if col not in ['ncodpers', 'int_date'] else col for col in trn.columns]
# 原本データとlagデータをncodperとint_dateを基準として合わせます。
# lagデータのint_dateは1だけなので、前の月の商品情報が挿入されます。
df_trn = trn.merge(trn_lag, on=['ncodpers','int_date'], how='left')
# メモリの効率化のため、不必要な変数をメモリから除去します。
del trn, trn_lag
# 前の月の商品情報が存在しない場合に備え、0に置き換えます。
for prod in prods:
    prev = prod + '_prev'
    df_trn[prev].fillna(0, inplace=True)
# 原本データで商品を保有しているかどうか、-lagデータで商品を保有しているかどうかを比較し、
# 新規購買変数paddを求めます。
for prod in prods:
    padd = prod + '_add'
    prev = prod + '_prev'
    df_trn[padd] = ((df_trn[prod] == 1) & (df_trn[prev] == 0)).astype(np.int8)
# 新規購買変数だけを抽出し、labelsに保存します。
add_cols = [prod + '_add' for prod in prods]
labels = df_trn[add_cols].copy()
labels.columns = prods
labels.to_csv('../input/labels.csv', index=False)

# 12. 月別新規購買データを累積棒グラフで可視化します。（コード2-13）
#labels = pd.read_csv('../input/labels.csv').astype(int)
fecha_dato = trn['fecha_dato']
labels['date'] = fecha_dato.fecha_dato
months = np.unique(fecha_dato.fecha_dato).tolist()
label_cols = labels.columns.tolist()[:24]
label_over_time = []
for i in range(len(label_cols)):
    label_over_time.append(labels.groupby(['date'])[label_cols[i]].agg('sum').tolist())
label_sum_over_time = []
for i in range(len(label_cols)):
    label_sum_over_time.append(np.asarray(label_over_time[i:]).sum(axis=0))
color_list = ['#F5B7B1','#D2B4DE','#AED6F1','#A2D9CE','#ABEBC6','#F9E79F','#F5CBA7','#CCD1D1']
f, ax = plt.subplots(figsize=(30, 15))
for i in range(len(label_cols)):
    sns.barplot(x=months, y=label_sum_over_time[i], color = color_list[i%8], alpha=0.7)
plt.legend([plt.Rectangle((0,0),1,1,fc=color_list[i%8], edgecolor = 'none') for i in range(len(label_cols))], label_cols, loc=1, ncol = 2, prop={'size':16})

# 13. 月別新規購買データの累積棒グラフを絶対値ではなく相対値で可視化（コード2－14）
label_sum_percent = (label_sum_over_time / (1.*np.asarray(label_sum_over_time).max(axis=0))) * 100
f, ax = plt.subplots(figsize=(30, 15))
for i in range(len(label_cols)):
    sns.barplot(x=months, y=label_sum_percent[i], color = color_list[i%8], alpha=0.7)
plt.legend([plt.Rectangle((0,0),1,1,fc=color_list[i%8], edgecolor = 'none') for i in range(len(label_cols))], \
           label_cols, loc=1, ncol = 2, prop={'size':16})
