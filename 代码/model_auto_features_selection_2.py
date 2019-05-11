# -*- coding: utf-8 -*-
###########################################################
# 注意提交前删除
import pandas as pd 
model_sample = pd.read_csv("model_sample.csv")
verify_sample = pd.read_csv("verify_sample.csv")

###########################################################
# 提交的代码
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score
import time 

###########################################################
# lgb params
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


model_sample['flag'] = 'train'
verify_sample['flag'] = 'test'

time_begin = time.time()
CV_NUM=5
###########################################################
if 'y' not in verify_sample.columns:
	print('verify_sample : ', -1)
	verify_sample['y'] = -1
###########################################################
# deal data
data = pd.concat([model_sample,verify_sample])
del model_sample, verify_sample
col = [col for col in data.columns if col not in ['user_id','y','flag']]

# norm
def max_min(data,col,epsilon=1e-5,fillna=0):
	data_norm = data.copy()
	for col_i in col:
		data_norm[col_i] = (data[col_i]-data[col_i].min())/(data[col_i].max()-data[col_i].min()+epsilon)
	data_norm = data_norm.fillna(fillna)
	return data_norm

def mean_std(data,col,epsilon=1e-5,fillna=0):
	data_norm = data.copy()
	for col_i in col:
		data_norm[col_i] = (data[col_i]-data[col_i].mean())/(data[col_i].std()+epsilon)
	data_norm = data_norm.fillna(fillna)
	return data_norm

data_norm = max_min(data,col)
del data

# corr
def corr_(x,y):
    return np.abs(np.corrcoef(x,y)[0][1])

def corr_col(data_norm,col,y='y',percentile=0.25):
	col_corr = []
	for col in col:
		col_corr.append(corr_(data_norm[col],data_norm[y]))
	df_corr = pd.DataFrame({'col':col,'corr':col_corr}).dropna()
	col_corr_ = df_corr['corr']
	col_corr_percentile = np.percentile(col_corr_,percentile)
	return df_corr, col_corr_, col_corr_percentile

# _,_,col_corr_percentile=corr_col(data_norm,col)

col_corr_percentile = 0.06

# make features
print("data_make_features begin")
epsilon=1e-5
func_dict = {
			'add': lambda x,y: x+y,
			'mins': lambda x,y: x-y,
			'div': lambda x,y: x/(y+epsilon),
			'multi': lambda x,y: x*y
			}

def func_make(data_norm,func_dict,threshold,col,y='y'):
	data_train = data_norm[data_norm[y]!=-1]
	data_make_features = data_norm[['user_id','y']].copy()
	for col_i in col:
		for col_j in col:
			for func_name, func in func_dict.items():
				func_features = func(data_train[col_i],data_train[col_j])
				if corr_(func_features,data_train[y]) > threshold:
					col_func_features = '-'.join([col_i,func_name,col_j])
					print(col_func_features)
					data_make_features[col_func_features] = func(data_norm[col_i],data_norm[col_j])
	return data_make_features

data_make_features = func_make(data_norm,func_dict,col_corr_percentile,col[:5])
print("data_make_features over")
print("features nums : ", len(data_make_features.columns))
print(data_make_features.columns)


# transform feantures
print("data_make_features2 begin")
epsilon=1e-5
col_cat = ['x_00'+str(i) for i in range(1,10)] + ['x_0'+str(i) for i in range(10,20)]
col_float = [col_ for col_ in col if col_ not in col_cat+['user_id','y','flag']]

transform_dict = {
			#'max_min': lambda x: (x-np.min(x))/(np.max(x)-np.min(x)),
			#'mean_std': lambda x: (x-np.mean(x))/(np.std(x)+epsilon),
			'abs_mean_std': lambda x: np.abs((x-np.mean(x))/(np.std(x)+epsilon)),
			#'nsigma': lambda x: np.round((x-np.mean(x))/(np.std(x)+epsilon))
			}

def groupby_transform(data_norm,col_cat,col_float,transform_dict,threshold,y='y'):
	data_train = data_norm[data_norm[y]!=-1]
	data_make_features = data_norm[['user_id','y']].copy()
	for col_cat_i in col_cat:
		for func_name, func in transform_dict.items():
			groupby_transform = data_train.groupby([col_cat_i]).transform(func)
			data_norm_groupby_transform = data_norm.groupby([col_cat_i]).transform(func)
			for col_float_i in col_float:
				transfrom_features = groupby_transform[col_float_i]
				if corr_(transfrom_features,data_train[y]) > threshold:
					col_func_features = '-'.join([col_cat_i,func_name,col_float_i])
					print(col_func_features)
					data_make_features[col_func_features] = data_norm_groupby_transform[col_float_i]
	return data_make_features

data_make_features2 = groupby_transform(data_norm,col_cat[:5],col_float[:5],transform_dict,col_corr_percentile)
print("data_make_features2 over")
print("features2 nums : ", len(data_make_features2.columns))
print(data_make_features2.columns)

# features 
del data_make_features['user_id'];del data_make_features['y']
del data_make_features2['user_id'];del data_make_features2['y']
data_features = pd.concat([data_norm,data_make_features,data_make_features2])
del data_norm; del data_make_features; del data_make_features2

data_features.replace([np.inf, -np.inf], np.nan)
data_features.fillna(0,inplace=True)



###################################################################################
# model test 
print("model test begin")
features_col = [col for col in data_features.columns if col not in ['user_id','y','flag']]
X = data_features[data_features['flag']=='train'][features_col].values
y = data_features[data_features['flag']=='train']['y'].values
X_pred = data_features[data_features['flag']=='test'][features_col].values


verify_sample = data_features[data_features['flag']=='test']

###################################################################################
# stacking
skf = StratifiedKFold(n_splits=CV_NUM, shuffle=True, random_state=1)
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

features_col = X.columns
X = X.values
X_pred = X_pred.values


###################################################################################
# selection features ####  to much time !!!
# print("features selection")
# from sklearn.ensemble import RandomForestClassifier
# clf =RandomForestClassifier(n_estimators=300,n_jobs=-1)
# selector = RFECV(estimator=clf, step=1, cv=StratifiedKFold(3), scoring='roc_auc')
# selector.fit(X, y)
 
# print(' Optimal number of features: %d' % selector.n_features_)
# sel_features = [f for f, s in zip(features_col, selector.support_) if s]
###################################################################################
# features import selection 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
	
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

model = lgb.train(params,
				lgb_train,
				num_boost_round=10000,
				valid_sets=lgb_eval,
				early_stopping_rounds=200)

features_import = pd.DataFrame({'features_import':model.feature_importance(),'features_col':features_col}).sort_values('features_import',ascending=False)

col_num = [200]
for i, num in enumerate(col_num):
	sel_features = features_import.head(num)['features_col'].tolist()

	X = data_features[data_features['flag']=='train'][sel_features]
	y = data_features[data_features['flag']=='train']['y']
	X_pred = data_features[data_features['flag']=='test'][sel_features]
	###################################################################################

	for cv in range(CV_NUM):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=np.random.randint(1000))
		
		# create dataset for lightgbm
		lgb_train = lgb.Dataset(X_train, y_train)
		lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

		# specify your configurations as a dict

		# train
		gbm = lgb.train(params,
					lgb_train,
					num_boost_round=10000,
					valid_sets=lgb_eval,
					early_stopping_rounds=200)

		y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
		pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
		if cv == 0:
			pred_out=pred
			ff = f1_score(y_test, np.where(y_pred >= 0.205,1,0))
		else:
			pred_out+=pred
			ff += f1_score(y_test, np.where(y_pred >= 0.205,1,0))
	if i == 0:
		pred_out_new = pred_out * 1.0 / CV_NUM
	else:
		pred_out_new += pred_out * 1.0 / CV_NUM


pred_out = pred_out_new / len(col_num)
pred = np.where(pred_out >= 0.205,1,0)
verify_sample['y_prediction'] = pred
predict_result = verify_sample[['user_id','y_prediction']]
###########################################################
time_over = time.time()
print("spend time : ", (time_over - time_begin) / 60.0)

# print(verify_sample['y'])
# print(pred)

###########################################################
# 非官方部分，线下测试
from sklearn.metrics import f1_score
print(f1_score(verify_sample['y'],pred))
###########################################################

###########################################################
# 官方部分
# f1_score = score(predict_result)
###########################################################


