# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
X_train = train.drop(['class'], axis=1)
Y_train = train['class']
X_test = test.drop(['class'], axis=1)