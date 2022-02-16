# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

# データの読み込み
train = pd.read_csv('../data/input/train.tsv', sep='\t')
test = pd.read_csv('../data/input/test.tsv', sep='\t')


'''
特徴量エンジニアリング
'''

# train['class']を数字に置き換え
class_dict = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3,
}

train['class']= train['class'].map(class_dict)


# 学習データとテストデータの連結
df = pd.concat([train, test], sort=False).reset_index(drop=True)


'''One-Hot Encoding'''

# カラムを指定
columns=['buying','maint','doors','persons','lug_boot','safety']

df = pd.get_dummies(df, columns=columns, drop_first=True)


'''前処理済みtrainとtestに再分割'''

preprocessed_train = df[~df['class'].isnull()]
preprocessed_test = df[df['class'].isnull()]


'''前処理済みtrainとtestを保存'''

preprocessed_train.to_csv('../data/processed/preprocessed_train.csv', header=True, index=False)
preprocessed_test.to_csv('../data/processed/preprocessed_test.csv', header=True, index=False)