#this module creates datasets and trains models
import re
import numpy as np
import csv
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPRegressor, MLPClassifier

modeldict={'lasso':Lasso, 'ridge':Ridge, 'logreg':LogisticRegression,
 'rfreg':RandomForestRegressor, 'rfclass': RandomForestClassifier, 'adabreg':AdaBoostRegressor, 
 'adabclass':AdaBoostClassifier, 'knnreg':KNeighborsRegressor, 'knnclass':KNeighborsClassifier, 
 'svreg':SVR,'svclass':SVC, 'mlpreg':MLPRegressor,'mlpclass':MLPClassifier}


def scores(y_ground_arr,y_pred_arr,class_thresholds):
	scores_arr=[0,0,0,0]
	for y_ground_pred in zip(y_pred_arr,y_ground_arr):
		temp=2*np.ceil(y_ground_pred[0]-class_thresholds)-y_ground_pred[1]
		for jj in temp:
			scores_arr[int(jj)]+=1
	false_pos=scores_arr[2]
	false_neg=scores_arr[-1]
	true_pos=scores_arr[1]
	true_neg=scores_arr[0]
	if false_neg==0:
		recall=1.
	else:
		recall=true_pos/(true_pos+false_neg)
	if false_pos==0:
		precision=1.
	else:
		precision=true_pos/(true_pos+false_pos)
	if precision==0. and recall==0.:
		f1score=0.
	else:
		f1score=2*recall*precision/(recall+precision)
	if (false_pos+false_neg)==0:
		jaccard_index=1.
	else:
		jaccard_index=true_pos/(false_pos+false_neg+true_pos)
	return recall,precision,f1score,jaccard_index

class StockPrediction:

	def __init__(self,dataframes_list=[[],[]],gdeltcorp_list=False,model_list=[],sp500t=True):
		self.dataset_names=dataframes_list[0]
		self.dataset_list=dataframes_list[1]
		if gdeltcorp_list!=False:
			for gdeltcorp in gdeltcorp_list:
				var_name = [k for k,var in locals().items() if var is gdeltcorp][0]
				self.dataset_names+=[var_name+'_bow',var_name+'_tfidf']
				self.dataset_list+=[gdeltcorp.vect_corpus_bow,]
				self.dataset_list+=[gdeltcorp.vect_corpus_tfidf,]
		self.models=model_list
		self.testmode={}
		self.xdata={}
		self.ydata={}
		self.sp500=[]
		if sp500t:
			self.load_sp500()
		return

	def import_nlp_data(self,dataframes_list=[[],[]],gdeltcorp_list=False):
		self.dataset_names+=dataframes_list[0]
		self.dataset_list+=dataframes_list[1]
		if gdeltcorp!=False:
			for gdeltcorp in gdeltcorp_list:
				var_name = [k for k,var in locals().items() if var is gdeltcorp][0]
				self.dataset_names+=[var_name+'_bow',var_name+'_tfidf']
				self.dataset_list+=[gdeltcorp.vect_corpus_bow,]
				self.dataset_list+=[gdeltcorp.vect_corpus_tfidf,]
		return

	def add_model(self,model_list):
		self.models+=model_list
		return

	def load_sp500(self):
		sp500=[]
		with open('data/SP500am.csv','r') as mycsvfile:
			reader=csv.reader(mycsvfile)
			for row in reader:
				row[0]=re.sub('-','',row[0])
				sp500+=[row]
		self.sp500=sp500[:963][::-1]
		return

	def prepare_x(self,dataframe,days,sp):
		x_arr=[]
		j=0
		for i,tomorrow in enumerate(days[1:]):
			next_bizday=sp[j+1][0]
			if tomorrow==next_bizday:
				after_we=0.
				latest_bizday=sp[j][0]
				today=days[i]
				if latest_bizday!=today:
					after_we=1.
				x_arr+=[list(dataframe.iloc[i])+[after_we,float(sp[j][-1])]]
				j+=1
		return np.array(x_arr)

	def prepare_y(self,days,sp):
		y_arr=[]
		j=0
		for i,tomorrow in enumerate(days[1:]):
			next_bizday=sp[j+1][0]
			if tomorrow==next_bizday:
				y_arr+=[[float(sp[j+1][-1]),(np.sign(float(sp[j+1][-1])-float(sp[j][-1]))+1)/2.]]
				j+=1
		return np.array(y_arr)

	def prepare_data(self,dataset_id):
		if isinstance(dataset_id,int):
			index_df=dataset_id
			dataset_name=self.dataset_names[dataset_id]
		else:
			dataset_name=dataset_id
			index_df=self.dataset_names.index(dataset_name)
		dataframe=self.dataset_list[index_df]
		days=list(dataframe.index.values)
		sp500=self.sp500
		self.xdata[dataset_name]=self.prepare_x(dataframe,days,sp500)
		self.ydata[dataset_name]=self.prepare_y(days,sp500)
		self.testmode[dataset_name]=False
		return

	def trval_test_split(self,dataset_id,test_frac):
		if isinstance(dataset_id,int):
			dataset_name=self.dataset_names[dataset_id]
		else:
			dataset_name=dataset_id
		if self.testmode[dataset_name]==True:
			x=np.concatenate(self.xdata[dataset_name])
			y=np.concatenate(self.ydata[dataset_name])
		else:
			x=self.xdata[dataset_name]
			y=self.ydata[dataset_name]

		x_trval, x_test, y_trval, y_test = train_test_split(x,y,test_size=test_frac)

		self.xdata[dataset_name]=(x_trval,x_test)
		self.ydata[dataset_name]=(y_trval,y_test)
		self.testmode[dataset_name]=True
		return

	def kfold_val_reg(self,n_folds_val,dataset_id,regressor,parm,seed):
		regressor=modeldict[regressor]
		if isinstance(dataset_id,int):
			dataset_name=self.dataset_names[dataset_id]
		else:
			dataset_name=dataset_id
		x_trainval=self.xdata[dataset_name][0]
		y_trainval=self.xdata[dataset_name][0][:,0]

		kf_val = KFold(n_splits=n_folds_val,shuffle=True,random_state=seed)
		avg_rms_mod_val=0
		avg_rms_mod_train=0
		for train_index, val_index in kf_val.split(x_trainval):
			x_train, x_val = x_trainval[train_index], x_trainval[val_index]
			y_train, y_val = y_trainval[train_index], y_trainval[val_index]
			if regressor in {Lasso,Ridge}:
				model=regressor(alpha=parm)
			elif regressor in {RandomForestRegressor,}:
				model=regressor(n_estimators=parm[0],max_features=parm[1])
			elif regressor in {MLPRegressor,}:
				model=regressor(activation=parm[0],hidden_layer_sizes=parm[1])
			elif regressor in {AdaBoostRegressor,}:
				model=regressor(n_estimators=parm)
			elif regressor in {KNeighborsRegressor,}:
				model=regressor(n_neighbors=parm)  
			elif regressor in {SVR,}:
				model=regressor(C=parm[0],kernel=parm[1])  
			else:
				print('houston, we have a unknown model problem')
				return
			model.fit(x_train,y_train)
			avg_rms_mod_val+=np.sqrt((sum((model.predict(x_val)-y_val)**2)/len(y_val)))
			avg_rms_mod_train+=np.sqrt((sum((model.predict(x_train)-y_train)**2)/len(y_train)))
		avg_rms_mod_val=avg_rms_mod_val/n_folds_val
		avg_rms_mod_train=avg_rms_mod_train/n_folds_val
		print('avg_train_rmse:',avg_rms_mod_train,'avg_validation_rmse:',avg_rms_mod_val)
		return

	def kfold_test_reg(self,dataset_id,regressor,parm):
		if isinstance(dataset_id,int):
			dataset_name=self.dataset_names[dataset_id]
		else:
			dataset_name=dataset_id
		x_trainval=self.xdata[dataset_name][0]
		y_trainval=self.xdata[dataset_name][0][:,0]
		x_test=self.xdata[dataset_name][1]
		y_test=self.xdata[dataset_name][1][:,0]

		coeff=True
		if regressor in {Lasso,Ridge}:
			model=regressor(alpha=parm)
		elif regressor in {RandomForestRegressor,}:
			model=regressor(n_estimators=parm[0],max_features=parm[1],max_depth=parm[2])
			coeff=False
		elif regressor in {MLPRegressor,}:
			model=regressor(activation=parm[0],hidden_layer_sizes=parm[1])
			coeff=False
		elif regressor in {AdaBoostRegressor,}:
			model=regressor(n_estimators=parm)
			coeff=False
		elif regressor in {KNeighborsRegressor,}:
			model=regressor(n_neighbors=parm)  
			coeff=False
		elif regressor in {SVR,}:
			model=regressor(C=parm[0],kernel=parm[1])  
			coeff=False
		else:
			print('houston, we have a unknown model problem')        
			return
		model.fit(x_trainval,y_trainval)

		rms_mod_test=np.sqrt((sum((model.predict(x_test)-y_test)**2)/len(y_test)))
		rms_rand_test=np.sqrt((sum((x_test[:,-1]-y_test)**2)/len(y_test)))
		print('model_test_rmse:',rms_mod_test,'flat_test_rmse:',rms_rand_test)

		if coeff:
			return model.coef_
		return


	def kfold_val_class(self,classifier,parm,seed,thres):
		if isinstance(dataset_id,int):
			dataset_name=self.dataset_names[dataset_id]
		else:
			dataset_name=dataset_id
		x_trainval=self.self.xdata[dataset_name][0]
		y_trainval=self.self.xdata[dataset_name][0][:,1]

		kf_val = KFold(n_splits=n_folds_val,shuffle=True,random_state=seed)
		avg_scores_train=np.zeros(3)
		avg_scores_val=np.zeros(3)
		for train_index, val_index in kf_val.split(x_trainval):
			x_train, x_val = x_trainval[train_index], x_trainval[val_index]
			y_train, y_val = y_trainval[train_index], y_trainval[val_index]
			if classifier in {LogisticRegression,}:
				model=classifier(penalty=parm[0], C=parm[1])
			elif classifier in {RandomForestClassifier,}:
				model=classifier(n_estimators=parm[0],max_features=parm[1],max_depth=parm[2])
			elif classifier in {MLPClassifier,}:
				model=classifier(activation=parm[0],hidden_layer_sizes=parm[1])
			elif classifier in {AdaBoostClassifier,}:
				model=classifier(n_estimators=parm)
			elif classifier in {KNeighborsClassifier,}:
				model=classifier(n_neighbors=parm)  
			elif classifier in {SVC,}:
				model=classifier(C=parm[0],kernel=parm[1])  
			else:
				print('houston, we have a unknown model problem')
				return
			model.fit(x_train,y_train)
			avg_scores_train+=np.array(scores(y_train,model.predict(x_train),[thres])[:3])
			avg_scores_val+=np.array(scores(y_val,model.predict(x_val),[thres])[:3])

		avg_scores_train=avg_scores_train/n_folds_val    
		avg_scores_val=avg_scores_val/n_folds_val
		print('avg_train_rec,prec,F1:',list(avg_scores_train),'avg_validation_rec,prec,F1:',list(avg_scores_val))
		return

	def kfold_test_class(self,dataset_id,classifier,parm,thres):
		if isinstance(dataset_id,int):
			dataset_name=self.dataset_names[dataset_id]
		else:
			dataset_name=dataset_id
		x_trainval=self.self.xdata[dataset_name][0]
		y_trainval=self.self.xdata[dataset_name][0][:,0]
		x_test=self.self.xdata[dataset_name][1]
		y_test=self.self.xdata[dataset_name][1][:,0]

		coeff=True
		if classifier in {LogisticRegression,}:
			model=classifier(penalty=parm[0], C=parm[1])
		elif classifier in {RandomForestClassifier,}:
			model=classifier(n_estimators=parm[0],max_features=parm[1],max_depth=parm[2])
			coeff=False
		elif classifier in {MLPClassifier,}:
			model=classifier(activation=parm[0],hidden_layer_sizes=parm[1])
			coeff=False
		elif classifier in {AdaBoostClassifier,}:
			model=classifier(n_estimators=parm)
			coeff=False
		elif classifier in {KNeighborsClassifier,}:
			model=classifier(n_neighbors=parm)  
			coeff=False
		elif classifier in {SVC,}:
			model=classifier(C=parm[0],kernel=parm[1])  
			coeff=False
		else:
			print('houston, we have a unknown model problem')
			return
            
		model.fit(x_trainval,y_trainval)

		scores_test=np.array(scores(y_test,model.predict(x_test),[thres])[:3])
		print('test_rec,prec,F1:',list(scores_test))   
		if coeff:
			return model.coef_
		return






