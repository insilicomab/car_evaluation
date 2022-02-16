# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from statistics import mean

import lightgbm as lgb

'''初期設定'''

# モデル名
model_name = 'LightGBM'

PROCESSED_TRAIN_DATA_PATH = '../data/processed/preprocessed_train.csv'
PROCESSED_TEST_DATA_PATH = '../data/processed/preprocessed_test.csv'
SAMPLESUB_PATH = './data/input/sample_submit.csv'
SUB_PATH = f'./submit/{model_name}'

# データの読み込み
train = pd.read_csv(PROCESSED_TRAIN_DATA_PATH)
test = pd.read_csv(PROCESSED_TEST_DATA_PATH)

# 説明変数と目的変数を指定
X_train = train.drop(['class', 'id'], axis=1)
Y_train = train['class']
X_test = test.drop(['class', 'id'], axis=1)

'''
モデルの構築と評価
'''

# 分割する
folds = 10
skf = StratifiedKFold(n_splits=folds)

# ハイパーパラメータの設定
params = {
    # 多値分類問題
    'objective': 'multiclass',
    # クラス数
    'num_class': 4,
}

# 各foldごとに作成したモデルごとの予測値を保存
models = []
accs = []
oof = np.zeros(len(X_train))

for train_index, val_index in skf.split(X_train, Y_train):
    x_train = X_train.iloc[train_index]
    x_valid = X_train.iloc[val_index]
    y_train = Y_train.iloc[train_index]
    y_valid = Y_train.iloc[val_index]
    
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)    
    
    model = lgb.train(params,
                      lgb_train, 
                      valid_sets=lgb_eval, 
                      num_boost_round=200, # 学習回数の実行回数
                      early_stopping_rounds=20, # early_stoppingの判定基準
                      verbose_eval=10)
    
    y_pred = model.predict(x_valid, num_iteration=model.best_iteration)
    y_pred_max = np.argmax(y_pred, axis=1)  # 最尤と判断したクラスの値にする
    acc = accuracy_score(y_valid, y_pred_max)
    print(acc)
    
    models.append(model)
    accs.append(acc)
    
    # 混同行列の作成
    cm = confusion_matrix(y_valid, y_pred_max)
    
    # heatmapによる混同行列の可視化
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.show()
    
    
# 平均AUCを計算する
acc_mean = mean(accs)
print(acc_mean)
 
# 特徴量重要度の表示
for model in models:
    lgb.plot_importance(model, importance_type='gain',
                        max_num_features=20)
    

"""
予測精度：
0.968644747393745
"""