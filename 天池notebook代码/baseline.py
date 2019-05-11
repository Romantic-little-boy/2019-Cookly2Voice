# 查看数据文件目录  list datalab files
!ls datalab/

# 查看个人永久空间文件  list files in your permanent storage
!ls /home/myspace/

# 查看当前kernel下已安装的包  list packages
!pip list --format=columns

!ls datalab/7572

!head datalab/7572/model_sample.csv

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

CV_NUM=5

data_sample = pd.read_csv('./datalab/7572/model_sample.csv')
data_sample_len = len(data_sample)
train_size = int(0.8 * data_sample_len)

model_sample = data_sample[:train_size].fillna(-1)
verify_sample = data_sample[train_size:].fillna(-1)

features_col = [c for c in model_sample.columns if c not in ['user_id','y']]
X = model_sample[features_col]
y = model_sample['y']
X_pred = verify_sample[features_col]

for cv in range(CV_NUM):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=np.random.randint(1000))

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'task':'train',
        'boosting_type':'gbdt',
        'num_leaves': 31,
        'objective': 'binary', 
        'learning_rate': 0.05, 
        'bagging_freq': 2, 
        'max_bin':256,
        'num_threads': 32
    } 

    # train
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_eval,
                early_stopping_rounds=200)

    pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
    if cv == 0:
        pred_out=pred
    else:
        pred_out+=pred

pred_out = pred_out * 1.0 / CV_NUM
pred = np.where(pred_out >= 0.225,1,0)
verify_sample['y_prediction'] = pred
predict_result = verify_sample[['user_id','y_prediction']]

print(f1_score(verify_sample['y'],pred))

