# ライブラリのインポート
import pandas as pd
import numpy as np
from scipy import stats

# 予測データの読み込み
lgb = pd.read_csv('../submit/LightGBM_acc_0.968644747393745.csv', header=None)
xgb = pd.read_csv('../submit/LightGBM_acc_0.968644747393745.csv', header=None)
rf = pd.read_csv('../submit/RandomForest_acc_0.8656909917134457.csv', header=None)

# 予測データの結合
df = pd.concat([lgb[1], xgb[1], rf[1]],axis=1)

# ダミー変数化
df = df.replace(['unacc','acc','good','vgood'], [0,1,2,3])

# アンサンブル学習
ensemble_array = np.array(df).T
pred = stats.mode(ensemble_array)[0].T # 予測データリストのうち最頻値を算出し、行と列を入れ替え

'''
提出
'''

# 提出用サンプルの読み込み
sub = pd.read_csv('../data/input/sample_submit.csv', sep=',', header=None)

# 目的変数カラムの置き換え
sub[1] = pred

# ダミー変数をもとの変数に戻す
sub[1] = sub[1].replace([0,1,2,3],['unacc','acc','good','vgood'])

# ファイルのエクスポート
sub.to_csv('../submit/ensemble.csv', header=None, index=None)