# -*- coding: utf-8 -*-
###########################################################
# 注意提交前删除
import pandas as pd 
model_sample = pd.read_csv("model_sample.csv")
verify_sample = pd.read_csv("verify_sample.csv")
###########################################################

###########################################################
# 提交的代码
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

CV_NUM=3

###########################################################
model_sample = model_sample.fillna(-1)
verify_sample = verify_sample.fillna(-1)

features_col = [c for c in model_sample.columns if c not in ['user_id','y']]
X = model_sample[features_col].values
y = model_sample['y'].values
X_pred = verify_sample[features_col].values


skf = StratifiedKFold(n_splits=CV_NUM, shuffle=True, random_state=1)

params = {
		'task': 'train',
		'boosting_type': 'gbdt',
		'objective': 'binary',
		'metric': 'auc',
		'num_leaves': 9,
		'learning_rate': 0.03,
		'feature_fraction_seed': 2,
		'feature_fraction': 0.9,
		'bagging_fraction': 0.8,
		'bagging_freq': 5,
		'min_data': 20,
		'min_hessian': 1,
		'verbose': -1,
		}


layer_train = np.zeros((X.shape[0], CV_NUM))
test_pred = np.zeros((X_pred.shape[0], CV_NUM))

for k,(train_index, test_index) in enumerate(skf.split(X, y)):
	X_train = X[train_index]
	y_train = y[train_index]
	X_test = X[test_index]
	y_test = y[test_index]

	# create dataset for lightgbm
	lgb_train = lgb.Dataset(X_train, y_train)
	lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

	# train
	gbm = lgb.train(params,
				lgb_train,
				num_boost_round=10000,
				valid_sets=lgb_eval,
				early_stopping_rounds=200)

	pred_y = gbm.predict(X_test, num_iteration=gbm.best_iteration)
	layer_train[test_index, 1] = pred_y

	pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
	test_pred[:, k] = pred
layer_test = test_pred.mean(axis=1)


X = pd.DataFrame(X)
X['pre']=layer_train[:,1]
X_pred = pd.DataFrame(X_pred)
X_pred['pre'] = layer_test

X = X.values
X_pred = X_pred.values

for cv in range(CV_NUM):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=cv)

# for cv,(train_index, test_index) in enumerate(skf.split(X, y)):
#	X_train = X[train_index]
#	y_train = y[train_index]
#	X_test = X[test_index]
#	y_test = y[test_index]
	
	# create dataset for lightgbm
	lgb_train = lgb.Dataset(X_train, y_train)
	lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

	# specify your configurations as a dict
	params = {
			'task': 'train',
			'boosting_type': 'gbdt',
			'objective': 'binary',
			'metric': 'auc',
			'num_leaves': 9,
			'learning_rate': 0.03,
			'feature_fraction_seed': 2,
			'feature_fraction': 0.9,
			'bagging_fraction': 0.8,
			'bagging_freq': 5,
			'min_data': 20,
			'min_hessian': 1,
			'verbose': -1,
			}

	# train
	gbm = lgb.train(params,
				lgb_train,
				num_boost_round=10000,
				valid_sets=lgb_eval,
				early_stopping_rounds=200)

	# print('Save model...')
	# save model to file
	# gbm.save_model('model.txt')

	# print('Start predicting...')
	# predict
	# y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
	# eval
	# print('The f1 of prediction is:', f1_score(y_test, np.where(y_pred >= 0.225,1,0)))

	pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
	if cv == 0:
		pred_out=pred
	else:
		pred_out+=pred


pred_out = pred_out * 1.0 / CV_NUM
pred = np.where(pred_out >= 0.24,1,0)
verify_sample['y_prediction'] = pred
predict_result = verify_sample[['user_id','y_prediction']]
###########################################################

###########################################################
# 非官方部分，线下测试
from sklearn.metrics import f1_score
print(f1_score(verify_sample['y'],pred))
###########################################################

###########################################################
# 官方部分
# f1_score = score(predict_result)
###########################################################


