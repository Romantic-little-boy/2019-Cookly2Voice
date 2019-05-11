# -*- coding: utf-8 -*-
###########################################################
# 提交的代码
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

class SBBTree():
	"""Stacking,Bootstap,Bagging----SBBTree"""
	""" author：Cookly """
	def __init__(self, params, stacking_num, bagging_num, bagging_test_size, num_boost_round, early_stopping_rounds):
		"""
		Initializes the SBBTree.
        Args:
          params : lgb params.
          stacking_num : k_flod stacking.
          bagging_num : bootstrap num.
          bagging_test_size : bootstrap sample rate.
          num_boost_round : boost num.
		  early_stopping_rounds : early_stopping_rounds.
        """
		self.params = params
		self.stacking_num = stacking_num
		self.bagging_num = bagging_num
		self.bagging_test_size = bagging_test_size
		self.num_boost_round = num_boost_round
		self.early_stopping_rounds = early_stopping_rounds

		self.model = lgb
		self.stacking_model = []
		self.bagging_model = []

	def fit(self, X, y):
		""" fit model. """
		if self.stacking_num > 1:
			layer_train = np.zeros((X.shape[0], 2))
			self.SK = StratifiedKFold(n_splits=self.stacking_num, shuffle=True, random_state=1)
			for k,(train_index, test_index) in enumerate(self.SK.split(X, y)):
				X_train = X[train_index]
				y_train = y[train_index]
				X_test = X[test_index]
				y_test = y[test_index]

				lgb_train = lgb.Dataset(X_train, y_train)
				lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

				gbm = lgb.train(self.params,
							lgb_train,
							num_boost_round=self.num_boost_round,
							valid_sets=lgb_eval,
							early_stopping_rounds=self.early_stopping_rounds)

				self.stacking_model.append(gbm)

				pred_y = gbm.predict(X_test, num_iteration=gbm.best_iteration)
				layer_train[test_index, 1] = pred_y

			X = np.hstack((X, layer_train[:,1].reshape((-1,1)))) 
		else:
			pass
		for bn in range(self.bagging_num):
			X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.bagging_test_size, random_state=bn)
	
			lgb_train = lgb.Dataset(X_train, y_train)
			lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

			gbm = lgb.train(params,
						lgb_train,
						num_boost_round=10000,
						valid_sets=lgb_eval,
						early_stopping_rounds=200)

			self.bagging_model.append(gbm)
		
	def predict(self, X_pred):
		""" predict test data. """
		if self.stacking_num > 1:
			test_pred = np.zeros((X_pred.shape[0], self.stacking_num))
			for sn,gbm in enumerate(self.stacking_model):
				pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
				test_pred[:, sn] = pred
			X_pred = np.hstack((X_pred, test_pred.mean(axis=1).reshape((-1,1))))  
		else:
			pass 
		for bn,gbm in enumerate(self.bagging_model):
			pred = gbm.predict(X_pred, num_iteration=gbm.best_iteration)
			if bn == 0:
				pred_out=pred
			else:
				pred_out+=pred
		return pred_out/self.bagging_num

###########################################################################
# test code
from sklearn.datasets import make_classification
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_gaussian_quantiles
from sklearn import metrics
from sklearn.metrics import f1_score
# X, y = make_classification(n_samples=1000, n_features=25, n_clusters_per_class=1, n_informative=15, random_state=1)
X, y = make_gaussian_quantiles(mean=None, cov=1.0, n_samples=1000, n_features=50, n_classes=2, shuffle=True, random_state=2)
# data = load_breast_cancer()
# X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
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
		'silent': 0
		}
# test 1
model = SBBTree(params=params, stacking_num=2, bagging_num=1,  bagging_test_size=0.33, num_boost_round=10000, early_stopping_rounds=200)
model.fit(X,y)
X_pred = X[0].reshape((1,-1))
pred=model.predict(X_pred)
print('pred')
print(pred)
print('TEST 1 ok')

'''
# test 1
model = SBBTree(params, stacking_num=1, bagging_num=1, bagging_test_size=0.33, num_boost_round=10000, early_stopping_rounds=200)
model.fit(X_train,y_train)
pred1=model.predict(X_test)

# test 2 
model = SBBTree(params, stacking_num=1, bagging_num=3, bagging_test_size=0.33, num_boost_round=10000, early_stopping_rounds=200)
model.fit(X_train,y_train)
pred2=model.predict(X_test)

# test 3 
model = SBBTree(params, stacking_num=5, bagging_num=1, bagging_test_size=0.33, num_boost_round=10000, early_stopping_rounds=200)
model.fit(X_train,y_train)
pred3=model.predict(X_test)

# test 4 
model = SBBTree(params, stacking_num=5, bagging_num=3, bagging_test_size=0.33, num_boost_round=10000, early_stopping_rounds=200)
model.fit(X_train,y_train)
pred4=model.predict(X_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test+1, pred1, pos_label=2)
print('auc: ',metrics.auc(fpr, tpr))

fpr, tpr, thresholds = metrics.roc_curve(y_test+1, pred2, pos_label=2)
print('auc: ',metrics.auc(fpr, tpr))

fpr, tpr, thresholds = metrics.roc_curve(y_test+1, pred3, pos_label=2)
print('auc: ',metrics.auc(fpr, tpr))

fpr, tpr, thresholds = metrics.roc_curve(y_test+1, pred4, pos_label=2)
print('auc: ',metrics.auc(fpr, tpr))
'''


