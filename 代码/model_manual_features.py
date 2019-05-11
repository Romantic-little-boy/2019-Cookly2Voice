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

eps = 1e-5
###########################################################
# features 1
col_info = ['x_00'+str(i) for i in range(3,10)]+['x_0'+str(i) for i in range(10,20)]
model_sample['x_3_19_sum'] = model_sample[col_info].sum()
verify_sample['x_3_19_sum'] = model_sample[col_info].sum()
###########################################################
# features 2
col_info = ['x_022','x_023','x_024','x_025','x_026']
for col in col_info:
	model_sample['rate_'+col+'_x_020'] = model_sample[col] / (eps + model_sample['x_020'])
	verify_sample['rate_'+col+'_x_020'] = verify_sample[col] / (eps + verify_sample['x_020'])

col_info = ['x_028','x_029','x_030','x_031','x_032']
for col in col_info:
	model_sample['rate_'+col+'_x_021'] = model_sample[col] / (eps + model_sample['x_021'])
	verify_sample['rate_'+col+'_x_021'] = verify_sample[col] / (eps + verify_sample['x_021'])
###########################################################
# features 3
col_info = ['x_020','x_021']
model_sample['x_20_21_sum'] = model_sample[col_info].sum()
verify_sample['x_20_21_sum'] = model_sample[col_info].sum()
###########################################################
# features 4
col_info = ['x_034','x_035','x_036','x_037','x_038','x_039','x_040']
model_sample['x_34_40_sum'] = model_sample[col_info].sum()
verify_sample['x_34_40_sum'] = model_sample[col_info].sum()
###########################################################
# features 4
col_info = ['x_034','x_035','x_036','x_037','x_038','x_039','x_040']
for col in col_info:
	model_sample['rate_'+col+'_x_34_40_sum'] = model_sample[col] / (eps + model_sample['x_34_40_sum'])
	verify_sample['rate_'+col+'_x_34_40_sum'] = verify_sample[col] / (eps + verify_sample['x_34_40_sum'])
###########################################################
# features 5 std
col_info = [['x_043','x_044'],['x_046','x_047'],['x_050','x_051'],['x_053','x_054'],\
['x_057','x_058'],['x_060','x_061'],['x_076','x_077'],['x_079','x_080'],\
['x_083','x_084'],['x_086','x_087'],['x_090','x_091'],['x_094','x_095'],\
['x_098','x_099'],['x_123','x_124'],['x_126','x_127']]
for col in col_info:
	model_sample['std_'+col[0]+'_'+col[1]] = model_sample[col[0]] / (eps + model_sample[col[1]])
	verify_sample['std_'+col[0]+'_'+col[1]] = verify_sample[col[0]] / (eps + verify_sample[col[1]])
###########################################################
# features 5 rate
col_info = [['x_064','x_062'],['x_064','x_063'],['x_062','x_063'],\
['x_067','x_065'],['x_067','x_066'],['x_065','x_066'],\
['x_070','x_068'],['x_070','x_069'],['x_068','x_069'],\
['x_073','x_071'],['x_073','x_072'],['x_072','x_071'],\
['x_101','x_100'],\
['x_104','x_102'],['x_104','x_103'],['x_103','x_102'],\
['x_108','x_106'],['x_108','x_107'],['x_107','x_106'],\
['x_111','x_109'],['x_111','x_110'],['x_110','x_109'],\
['x_114','x_112'],['x_114','x_113'],['x_113','x_112'],\
['x_117','x_115'],['x_117','x_116'],['x_116','x_115'],\
['x_120','x_118'],['x_120','x_119'],['x_119','x_118'],\
['x_130','x_128'],['x_130','x_129'],['x_129','x_128'],\
['x_133','x_134'],['x_133','x_132'],['x_134','x_132'],\
['x_138','x_139'],['x_138','x_137'],['x_139','x_137'],\
['x_143','x_144'],['x_143','x_142'],['x_144','x_142'],\
['x_151','x_149'],['x_152','x_149'],['x_152','x_151'],\
['x_154','x_153'],['x_156','x_153'],['x_157','x_153'],\
['x_158','x_153'],['x_159','x_153'],['x_154','x_155'],\
['x_164','x_162'],['x_165','x_162'],['x_165','x_164'],\
['x_167','x_166'],['x_169','x_166'],['x_170','x_166'],\
['x_171','x_166'],['x_180','x_181'],['x_167','x_168'],\
['x_172','x_167'],['x_177','x_175'],['x_178','x_175'],\
['x_178','x_177'],['x_180','x_179'],['x_182','x_179'],\
['x_183','x_179'],['x_184','x_179'],['x_180','x_181'],\
['x_185','x_180'],['x_189','x_188'],['x_191','x_190'],\
['x_193','x_192'],['x_195','x_194'],['x_197','x_196'],\
['x_199','x_198'],['x_196','x_188'],['x_192','x_188']]
for col in col_info:
	model_sample['r_'+col[0]+'_'+col[1]] = model_sample[col[0]] / (eps + model_sample[col[1]])
	verify_sample['r_'+col[0]+'_'+col[1]] = verify_sample[col[0]] / (eps + verify_sample[col[1]])
###########################################################
# features 6 131 - 146
col_info = [['x_135','x_136'],['x_140','x_141'],['x_145','x_146']]
for col in col_info:
	model_sample['range_'+col[0]+'_'+col[1]] = model_sample[col[0]] - model_sample[col[1]]
	verify_sample['range_'+col[0]+'_'+col[1]] = verify_sample[col[0]] - verify_sample[col[1]]

###########################################################

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
	y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
	# eval
	# print('The f1 of prediction is:', f1_score(y_test, np.where(y_pred >= 0.225,1,0)))

	pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
	if cv == 0:
		pred_out=pred
		ff = f1_score(y_test, np.where(y_pred >= 0.3,1,0))
	else:
		pred_out+=pred
		ff += f1_score(y_test, np.where(y_pred >= 0.3,1,0))


pred_out = pred_out * 1.0 / CV_NUM
pred = np.where(pred_out >= 0.2,1,0)
verify_sample['y_prediction'] = pred
predict_result = verify_sample[['user_id','y_prediction']]
###########################################################

###########################################################
# 非官方部分，线下测试
from sklearn.metrics import f1_score
print(f1_score(verify_sample['y'],pred))
###########################################################
###########################################################

###########################################################
# 官方部分
# f1_score = score(predict_result)
###########################################################


