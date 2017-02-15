"""
This module creates datasets from input dataframes of (nlp) feature vectors and trains and validates
a variety of simple scikitlearn models on them. This module is in constant process of generalization.
"""
import re
import os
import numpy as np
import csv
import datetime
from itertools import product
import bisect
from sklearn.model_selection import train_test_split, KFold,TimeSeriesSplit
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import scale
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPRegressor, MLPClassifier

#this dictionary is used to input scikit learn ML models into the training and testing methods below
modeldict={'lasso':Lasso, 'ridge':Ridge, 'logreg':LogisticRegression,
 'rfreg':RandomForestRegressor, 'rfclass': RandomForestClassifier, 'adabreg':AdaBoostRegressor, 
 'adabclass':AdaBoostClassifier, 'knnreg':KNeighborsRegressor, 'knnclass':KNeighborsClassifier, 
 'svmreg':SVR,'svmclass':SVC, 'mlpreg':MLPRegressor,'mlpclass':MLPClassifier}


def scores(y_ground_arr,y_pred_arr,class_thresholds):
	"""
	This function explicitly computes some performance estimators of a binary classification algorithm
	starting from a list of ground-truth class values and one of predicted class scores, supplemented with
	a list of detection thresholds for the classes. Each data point should be a numpy array.
	I'll generalize this in the future.
	"""
	scores_arr=[0,0,0,0]
	#Careful here, this method isn't the easiest to read. Let me show a practical example
	#y_ground_pred=[[0,0,1,1,0],[0.45,0.1,0.99,0.3,0.6]] with thresholds [0.4,0.4,0.6,0.25,0.9]
	#then the predicted classes are [1,0,1,0,0] (so we have 1 false pos, 1 false neg, 2 true neg, 
	#1 true pos). So temp=2*[0,0,1,1,0]-[1,0,1,0,0]=[-1,0,1,2,0] which shows how false pos will be 
	#marked by a value -1, false neg by a 2, true pos by a 1 and true neg by a 0, which is why I do:
	#false_pos=scores_arr[2]
	#false_neg=scores_arr[-1]
	#true_pos=scores_arr[1]
	#true_neg=scores_arr[0]
	for y_ground_pred in zip(y_pred_arr,y_ground_arr):
		temp=2*np.ceil(y_ground_pred[0]-class_thresholds)-y_ground_pred[1]
		for jj in temp:
			scores_arr[int(jj)]+=1
	false_pos=scores_arr[2]
	false_neg=scores_arr[-1]
	true_pos=scores_arr[1]
	true_neg=scores_arr[0]
	#these all just follow from the textbook definitions
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

def nextday(today):
	"""
	This function computes the real-calendar next day from 'today'. The i/o format is 'yyyymmdd'.
	"""
	today=datetime.date(int(today[:4]),int(today[4:6]),int(today[6:]))
	return (today+datetime.timedelta(days=1)).strftime("%Y%m%d")

def search_space_builder(parm_ranges,graininess=10):
	search_space=[]
	for param in parm_ranges:
		param=param[1]
		if isinstance(param,list) and len(param)==3:
			interval=param[1]-param[0]
			new_space=[param[0]+interval*i/(graininess-1) for i in range(graininess)]
			search_space+=[new_space]
		elif isinstance(param,set):
			search_space+=[list(param)]
	return product(*search_space)

def new_parm_ranges(parm_ranges,graininess=10):
	new_parm_ranges=[]
	for param in parm_ranges:
		if isinstance(param[1],list) and len(param[1])==3:
			interval=param[1][1]-param[1][0]
			new_space=[param[0],[param[1][0]+interval*i/(graininess-1) for i in range(graininess)]]
			insertion_ind=bisect.bisect(new_space[1],param[1][2])
			if (insertion_ind in {0,len(new_space[1])}) or ((insertion_ind==1) and (param[1][2] in new_space[1])):
				new_parm_ranges+=[[param[0],[param[1][2]-interval/2,param[1][2]+interval/2,param[1][2]]]]
			elif insertion_ind in {1,len(new_space[1])-1}:
				new_parm_ranges+=[[param[0],[new_space[1][insertion_ind-1],new_space[1][insertion_ind],param[1][2]]]]
			else:
				new_parm_ranges+=[[param[0],[new_space[1][insertion_ind-1],new_space[1][insertion_ind+1],param[1][2]]]]
		elif isinstance(param[1],set):
			new_parm_ranges+=[param]
	return new_parm_ranges


def model_init_reg(regressor,parm):
	regressor=modeldict[regressor]
	if regressor in {Lasso,Ridge}:
		model=regressor(alpha=parm)
	elif regressor in {RandomForestRegressor,}:
		model=regressor(n_estimators=parm[0],max_features=parm[1],max_depth=parm[2])
	elif regressor in {MLPRegressor,}:
		model=regressor(activation=parm[0],hidden_layer_sizes=parm[1])
	elif regressor in {AdaBoostRegressor,}:
		model=regressor(n_estimators=parm[0])
	elif regressor in {KNeighborsRegressor,}:
		model=regressor(n_neighbors=parm[0])  
	elif regressor in {SVR,}:
		model=regressor(C=parm[0],kernel=parm[1])  
	else:
		print('houston, we have an unknown model problem')       
	return model

def model_init_class(classifier,parm):
	classifier=modeldict[classifier]
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
		print('houston, we have an unknown model problem')
	return model



class StockPrediction:
	"""
	This class creates datasets from input dataframes of (nlp) feature vectors and trains and
	 validates a variety of simple scikitlearn models on them. For generality and ease of model 
	 exploration, it stores all the processed data (x/ydata) as dictionaries of numpy arrays, ready 
	 to be fed to ML algorithms, also predictions (yhat_*) keeping track of test/validation stage 
	 (testmode) and in the future of model parameters and settings.
	"""
	def __init__(self,dataframes_list=[[],[],[]],gdeltcorp_list=False,sp500t=True,update=False,datadir='../data/'):
		self.datadir=datadir
		self.dataset_names=dataframes_list[0]
		self.dataset_list=dataframes_list[1]
		#list of used callable w2v models
		self.w2v_models={name_model[0]:name_model[1] for name_model in zip(dataframes_list[0],dataframes_list[2])}
		if gdeltcorp_list!=False:
			for gdeltcorp in gdeltcorp_list:
				var_name = [k for k,var in locals().items() if var is gdeltcorp][0]
				self.dataset_names+=[var_name+'_bow',var_name+'_tfidf',var_name+'_w2v']
				self.dataset_list+=[gdeltcorp.vect_corpus_bow,gdeltcorp.vect_corpus_tfidf,gdeltcorp.word2vec_corpus]
				self.w2v_models+=[gdeltcorp.w2vec_model]
		#have we already split the data with key 'something' into training+validation and testing?
		self._testmode={}
		#nlp+time series features for dataset of key 'something'
		self.xdata={}
		self.todaysx={}
		#target data for regression, incremental regression, and bullish/bearish classification
		self.ydata={}
		self.sp500={}
		#here I put the tested models for lookup
		self.models={}
		#yhat arrays are all going to host model predictions
		self.yhat_reg={}
		self.yhat_reg_diff={}
		self.yhat_class={}
		if sp500t:
			self._load_sp500(update=update)
		return

	def import_nlp_data(self,dataframes_list=[[],[],[]],gdeltcorp_list=False):
		"""
		This method essentially works in the same exact way as the initializer up above, 
		simply adding to the data already there. Please see above for comments.
		"""
		self.dataset_names+=dataframes_list[0]
		self.dataset_list+=dataframes_list[1]
		self.w2v_models={name_model[0]:name_model[1] for name_model in zip(dataframes_list[0],dataframes_list[2])}
		if gdeltcorp_list !=False:
			for gdeltcorp in gdeltcorp_list:
				var_name = [k for k,var in locals().items() if var is gdeltcorp][0]
				self.dataset_names+=[var_name+'_bow',var_name+'_tfidf',var_name+'_w2v']
				self.dataset_list+=[gdeltcorp.vect_corpus_bow,gdeltcorp.vect_corpus_tfidf,gdeltcorp.word2vec_corpus]
				self.w2v_models+=[gdeltcorp.w2vec_model]
		return

	def _load_sp500(self,update=False):
		"""
		This method loads S&P500 index data, probably it could be part of the initializer but for 
		generality it's separate. One could envision a future where other loaders load different 
		time series to predict.
		"""
		if update:
			os.system('wget -nv -O SP500am.csv http://chart.finance.yahoo.com/table.csv?s=^GSPC')
			os.system('mv SP500am.csv '+self.datadir+'SP500am.csv')
		with open(self.datadir+'SP500am.csv','r') as mycsvfile:
			reader=csv.reader(mycsvfile)
			temp_prev='first day'
			self.sp500[temp_prev]=[]
			for row in reader:
				today=re.sub('-','',row[0])
				self.sp500[today]=[temp_prev]+row[1:]
				self.sp500[temp_prev]+=[today]
				temp_prev=today
			self.sp500[temp_prev]+=['last day']
		del self.sp500['first day']
		del self.sp500['Date']
		return

	def _dataset_id_parser(self,dataset_id,index=False):

		if isinstance(dataset_id,int):
			if index:
				return self.dataset_names[dataset_id],dataset_id
			else:
				return self.dataset_names[dataset_id]
		elif isinstance(dataset_id,str):
			if index:
				return dataset_id, self.dataset_names.index(dataset_id)
			else:
				return dataset_id
		else:
			raise KeyError


	def prepare_data(self,dataset_id,today_pred=False):
		"""
		This method simply reads a certain dataframe labeled by a certain name or ordinal number (dataset_id)
		together with the targets y and converts this information into numpy arrays stored in dictionaries
		"""

		dataset_name, index_df =self._dataset_id_parser(dataset_id,index=True)

		dataframe=self.dataset_list[index_df]
		days=list(dataframe.index.values)
		if today_pred:
			self.todaysx[dataset_name]=self._prepare_x_today(dataframe)
		self.xdata[dataset_name]=self._prepare_x(dataframe,days)
		self.ydata[dataset_name]=self._prepare_y(days)
		self._testmode[dataset_name]=False 
		return


	def _prepare_x_today(self,dataframe):
		"""
		This method is used to convert today's data into numpy arrays containing the input features for prediction
		"""
		sp=self.sp500
		today=dataframe.index[-1]
		after_weekend=0.
		x_row=[list(dataframe.loc[today])+[after_weekend,float(sp[today][-2])]]
		return np.array(x_row)

	def _prepare_x(self,dataframe,days):
		"""
		This method is used to specifically convert datasets into numpy arrays containing the input features
		"""
		sp=self.sp500
		x_arr=[]
		for today in days:
			tomorrow=nextday(today)
			if tomorrow in sp.keys():
				#the after_weekend feature is a categorical one I engineered that is 1 if the predicted day
				#falls after a holiday or weekend
				after_weekend=0.
				if today!=sp[tomorrow][-1]:
					after_weekend=1.
				x_arr+=[list(dataframe.loc[today])+[after_weekend,float(sp[sp[tomorrow][-1]][-2])]]
		return np.array(x_arr)


	def _prepare_y(self,days):
		"""
		This method is used to specifically convert datasets into numpy arrays containing the output/target
		"""
		sp=self.sp500
		y_arr=[]
		for today in days:
			tomorrow=nextday(today)
			if tomorrow in sp.keys():
				#every data point has 3 possible targets so far: 
				#[tomorrow's S&P value, bullish=1/bearish=0,tomorrow's S&P - today's S&P]
				difference=float(sp[tomorrow][-2])-float(sp[sp[tomorrow][-1]][-2])
				y_arr+=[[float(sp[tomorrow][-2]),int((np.sign(difference)+1)/2.),difference]]
		return np.array(y_arr)


	def trval_test_split(self,dataset_id,test_frac,time_srs=True,null=False):
		"""
		This method splits the dataset labelled by 'dataset_id' into training+validation and
		test datasets in a ratio of test_frac
		"""
		dataset_name=self._dataset_id_parser(dataset_id)

		if self._testmode[dataset_name]==True:
			x=np.concatenate(self.xdata[dataset_name])
			y=np.concatenate(self.ydata[dataset_name])
		else:
			x=self.xdata[dataset_name]
			y=self.ydata[dataset_name]

		if null:
			# this is used for direct prediction, no testing
			x_trval=x
			x_test=[]
			y_trval=y
			y_test=[]
		elif time_srs:
			break_ind=int((1-test_frac)*len(x))
			x_trval=x[:break_ind]
			x_test=x[break_ind:]
			y_trval=y[:break_ind]
			y_test=y[break_ind:]
		else:
			#using scikit learn splitting function
			x_trval, x_test, y_trval, y_test = train_test_split(x,y,test_size=test_frac)

		#storing the data into the same dictionary entries they came from, notice that this creates 
		#a different format, because now each dict value is no longer an array but a tuple of two arrays:
		#(train+val,test). I keep track of this with the 'testmode' attribute
		self.xdata[dataset_name]=(x_trval,x_test)
		self.ydata[dataset_name]=(y_trval,y_test)
		self._testmode[dataset_name]=True #please take care of this inconsistency
		return

	def kfold_val_reg(self,n_folds_val,dataset_id,regressor,parm,seed,differential=False,scaling=False,time_srs=False,verbose=True):
		"""
		This method performs a k-fold validation of a regression model on a certain dataset labelled by 
		'dataset_id', with a certain regressor, with certain parameters 'parm', distinguishing between a
		full tomorrow's value prediction and a tomorrow's increment prediction (argument differential).
		Scaling of (all) the features is also available. For reproducibilty, I also have a random k-folding seed.
		Example input: kfold_val_reg(10,'apriltodectfidf','lasso',1.3,10,differential=True,scaling=True)
		"""
		
		dataset_name=self._dataset_id_parser(dataset_id)

		x_trainval=self._xretrieval(dataset_name,scaling=scaling)[0]

		if differential:
			y_trainval=self.ydata[dataset_name][0][:,2]
		else:
			y_trainval=self.ydata[dataset_name][0][:,0]
		
		if time_srs:
			kf_val = TimeSeriesSplit(n_splits=n_folds_val)
		else:
			#the seed comes in here
			kf_val = KFold(n_splits=n_folds_val,shuffle=True,random_state=seed)

		avg_rms_mod_val=0
		avg_rms_mod_train=0
		#splitting validation from training, scaling is optional
		for train_index, val_index in kf_val.split(x_trainval):
			x_train, x_val = x_trainval[train_index], x_trainval[val_index]
			y_train, y_val = y_trainval[train_index], y_trainval[val_index]
			#parsing parameters and initializing model, given the chosen regressor
			model=model_init_reg(regressor,parm)

			model.fit(x_train,y_train)
			#storing the train/validation rmse performances
			avg_rms_mod_val+=np.sqrt((sum((model.predict(x_val)-y_val)**2)/len(y_val)))
			avg_rms_mod_train+=np.sqrt((sum((model.predict(x_train)-y_train)**2)/len(y_train)))
		#averaging over the several folds
		avg_rms_mod_val=avg_rms_mod_val/n_folds_val
		avg_rms_mod_train=avg_rms_mod_train/n_folds_val
		if verbose:
			#printing the average train/validation rmse performances
			print('avg_train_rmse:',avg_rms_mod_train,'avg_validation_rmse:',avg_rms_mod_val)
		return avg_rms_mod_val


	def _xretrieval(self,dataset_name,scaling):
		if scaling:
			x_trainval=self.xdata[dataset_name][0]
			x_test=self.xdata[dataset_name][1]
			scale(x_trainval)
			scale(x_test)
		else:
			x_trainval=self.xdata[dataset_name][0]
			x_test=self.xdata[dataset_name][1]
		return x_trainval, x_test


	def kfold_test_reg(self,dataset_id,regressor,parm,differential=False,scaling=False,verbose=True):
		"""
		Once a model is finally picked, given its performance on the validation sets (model tuning),
		we should see how it performs on the never-seen-before test set. This is what the method does.
		"""
		dataset_name=self._dataset_id_parser(dataset_id)

		x_trainval, x_test = self._xretrieval(dataset_name,scaling=scaling)

		#for regression, this is my benchmark model, that next day's prediction is equal to today's closing
		if not differential:
			y_flat=x_test[:,-1] #doing this before the scaling takes place

		if differential:
			y_trainval=self.ydata[dataset_name][0][:,2]
			y_test=self.ydata[dataset_name][1][:,2]
		else:
			y_trainval=self.ydata[dataset_name][0][:,0]
			y_test=self.ydata[dataset_name][1][:,0]

		model=model_init_reg(regressor,parm)
		model.fit(x_trainval,y_trainval)

		rms_mod_test=np.sqrt((sum((model.predict(x_test)-y_test)**2)/len(y_test)))

		# if we are perfoming tomorrow-today prediction, then my benchmark model is a prediction of the average increase on the training set...
		if differential:
			flat_model=np.mean(y_trainval)
			self.yhat_reg_diff[dataset_name]=model.predict(x_test)		
		# ...otherwise a flat prediction	
		else:
			flat_model=y_flat+np.mean(y_trainval-x_trainval[:,-1])
			self.yhat_reg[dataset_name]=model.predict(x_test)			
		rms_rand_test=np.sqrt((sum((flat_model-y_test)**2)/len(y_test)))
		#aaand these are the rmse performances of the model vs the benchmark
		if verbose:
			print('model_test_rmse:',rms_mod_test,'flat_test_rmse:',rms_rand_test)

		#here I store the model
		self.models[dataset_name]=model
		return rms_mod_test,rms_rand_test


	def auto_ts_val_test_reg(self,dataset_id,regressor,parm_ranges,parm_search_iter=5,n_folds_val=10,n_folds_test=5,scaling=False,differential=False,one_shot=True,verbose=False,notest=False,seed=100):
		"""
		This method performs an automated training validation, tuning and testing.
		"""
		dataset_name=self._dataset_id_parser(dataset_id)
		
		if notest:
			self.prepare_data(dataset_name,today_pred=True)
		else:
			self.prepare_data(dataset_name,today_pred=False)
	

		x=self.xdata[dataset_name]
		y=self.ydata[dataset_name]
		numb_datapoints=len(y)

		if notest:
			n_folds_test=0

		if n_folds_test>numb_datapoints-2:
			print('Please choose a smaller maximum test fraction, not enough data...')
			return

		if notest:
			x_trval=x
			y_trval=y
			x_test=[]
			y_test=[]
			self.xdata[dataset_name]=(x_trval,x_test)
			self.ydata[dataset_name]=(y_trval,y_test)
			self._testmode[dataset_name]=True

			min_err=10**9
			val_fold=0
			any_change=True
			while val_fold <parm_search_iter and (any_change or val_fold<5):
				any_change=False
				current_search_space=search_space_builder(parm_ranges,graininess=10)
				for parm in current_search_space:
					if verbose:
						print(parm)
					avg_rms_mod_val=self.kfold_val_reg(n_folds_val,dataset_id,regressor,list(parm),seed,differential=differential,scaling=scaling,time_srs=True,verbose=verbose)
					if min_err>avg_rms_mod_val:
						min_err=avg_rms_mod_val
						min_parm=parm
						any_change=True
				val_fold+=1
				for i,param in enumerate(parm_ranges):
					if isinstance(param[1],list) and len(param[1])==3:
						param[1][2]=min_parm[i]
				parm_ranges=new_parm_ranges(parm_ranges,graininess=10)
			model=model_init_reg(regressor,min_parm)
			x_trainval=x_trval
			if differential:
				y_trainval=y_trval[:,2]
			else:
				y_trainval=y_trval[:,0]
			model.fit(x_trainval,y_trainval)
			self.models[dataset_name]=model
			return model.predict(self.todaysx[dataset_name])

		break_indices=[numb_datapoints-n_folds_test+i for i in range(n_folds_test)]
		performances=[]
		for break_index in break_indices:
			if one_shot:
				#we only test on the very next day!!
				x_trval=x[:break_index]
				x_test=x[break_index:break_index+1]
				y_trval=y[:break_index]
				y_test=y[break_index:break_index+1]
			else:
				#we test on all following days
				x_trval=x[:break_index]
				x_test=x[break_index:]
				y_trval=y[:break_index]
				y_test=y[break_index:]

			self.xdata[dataset_name]=(x_trval,x_test)
			self.ydata[dataset_name]=(y_trval,y_test)
			self._testmode[dataset_name]=True

			min_err=10**9
			val_fold=0
			any_change=True
			while val_fold <parm_search_iter and (any_change or val_fold<5):
				any_change=False
				current_search_space=search_space_builder(parm_ranges,graininess=10)
				for parm in current_search_space:
					if verbose:
						print(parm)
					avg_rms_mod_val=self.kfold_val_reg(n_folds_val,dataset_id,regressor,list(parm),seed,differential=differential,scaling=scaling,time_srs=True,verbose=verbose)
					if min_err>avg_rms_mod_val:
						min_err=avg_rms_mod_val
						min_parm=parm
						any_change=True
				val_fold+=1
				for i,param in enumerate(parm_ranges):
					if isinstance(param[1],list) and len(param[1])==3:
						param[1][2]=min_parm[i]
				parm_ranges=new_parm_ranges(parm_ranges,graininess=10)
			
			#performances+=[[min_parm,min_err]]
			#print(performances[-1])
			res=self.kfold_test_reg(dataset_id,regressor,min_parm,differential=differential,scaling=scaling)
			performances+=[res]

		return list(zip(np.mean(performances,axis=0),np.std(performances,axis=0)))


	def kfold_val_class(self,n_folds_val,dataset_id,classifier,parm,seed,thres=0.5,scaling=False,time_srs=True,verbose=True):
		"""
		This method performs a k-fold validation of a classification model on a certain dataset labelled by 
		'dataset_id', with a certain classifier, with certain parameters 'parm'. A list of classification 
		thresholds has to be provided as well to feed the outside function 'scores' (see way above), but 
		in practice for now leave it to [0.5].
		For reproducibilty, I also have a random k-folding seed.
		Example input: kfold_val_reg(10,'apriltodectfidf','logreg',1.3,10,[0.5])
		Please see the regression methods above for comments.
		"""
		dataset_name=self._dataset_id_parser(dataset_id)

		y_trainval=self.ydata[dataset_name][0][:,1]

		x_trainval=self._xretrieval(dataset_name,scaling=scaling)[0]
		
		if time_srs:
			kf_val = TimeSeriesSplit(n_splits=n_folds_val)
		else:
			#the seed comes in here
			kf_val = KFold(n_splits=n_folds_val,shuffle=True,random_state=seed)
		
		avg_scores_train=np.zeros(3)
		avg_scores_val=np.zeros(3)
		for train_index, val_index in kf_val.split(x_trainval):
			x_train, x_val = x_trainval[train_index], x_trainval[val_index]
			y_train, y_val = y_trainval[train_index], y_trainval[val_index]

			model=model_init_class(classifier,parm)
			model.fit(x_train,y_train)
			#the difference with the regressor is that the performance is evaluated with prec, reg and f1
			avg_scores_train+=np.array(scores(y_train,model.predict(x_train),[thres])[:3])
			avg_scores_val+=np.array(scores(y_val,model.predict(x_val),[thres])[:3])

		avg_scores_train=avg_scores_train/n_folds_val    
		avg_scores_val=avg_scores_val/n_folds_val
		if verbose:
			print('avg_train_rec,prec,F1:',list(avg_scores_train),'avg_validation_rec,prec,F1:',list(avg_scores_val))
		return avg_scores_val

	def kfold_test_class(self,dataset_id,classifier,parm,thres,scaling=False,verbose=True):
		"""
		Once a model is finally picked, given its performance on the validation sets (model tuning),
		we should see how it performs on the never-seen-before test set. This is what the method does.
		Please see the regression methods above for comments.
		"""

		dataset_name=self._dataset_id_parser(dataset_id)

		x_trainval, x_test = self._xretrieval(dataset_name,scaling=scaling)

		y_trainval=self.ydata[dataset_name][0][:,1]
		y_test=self.ydata[dataset_name][1][:,1]

		model=model_init_class(classifier,parm)

		model.fit(x_trainval,y_trainval)

		self.yhat_class[dataset_name]=model.predict(x_test)
		scores_test=np.array(scores(y_test,self.yhat_class[dataset_name],[thres])[:3])
		scores_flat=np.array(scores(y_test,np.array([sum(y_trainval)/len(y_trainval)]),[thres])[:3])
		#aaand these are the rmse performances of the model vs the benchmark
		if verbose:
			print('test_rec,prec,F1:',list(scores_test),'benchmark_rec,prec,F1:',list(scores_flat))
		
		#here I store the model
		self.models[dataset_name]=model

		return [scores_test],[scores_flat]



	def auto_ts_val_test_class(self,dataset_id,classifier,parm_ranges,parm_search_iter=5,n_folds_val=10,n_folds_test=5,scaling=False,one_shot=True,verbose=False,notest=False,seed=100):
		"""
		This method performs an automated training validation, tuning and testing.
		"""
		dataset_name=self._dataset_id_parser(dataset_id)
		
		if notest:
			self.prepare_data(dataset_name,today_pred=True)
		else:
			self.prepare_data(dataset_name,today_pred=False)
	

		x=self.xdata[dataset_name]
		y=self.ydata[dataset_name]
		numb_datapoints=len(y)

		if notest:
			n_folds_test=0

		if n_folds_test>numb_datapoints-2:
			print('Please choose a smaller maximum test fraction, not enough data...')
			return

		if notest:
			x_trval=x
			y_trval=y
			x_test=[]
			y_test=[]
			self.xdata[dataset_name]=(x_trval,x_test)
			self.ydata[dataset_name]=(y_trval,y_test)
			self._testmode[dataset_name]=True

			max_score=0.
			val_fold=0
			any_change=True
			for parm_range in parm_ranges:
				if isinstance(parm_range[1],list):
					max_parm+=[parm_range[1][2]]
				elif isinstance(parm_range[1],set):
					max_parm+=[list(parm_range[1])[0]]
				else:
					raise KeyError
			while val_fold <parm_search_iter and (any_change or val_fold<5):
				any_change=False
				current_search_space=search_space_builder(parm_ranges,graininess=10)
				for parm in current_search_space:
					if verbose:
						print(parm)
					avg_scores_val=self.kfold_val_class(n_folds_val,dataset_id,classifier,list(parm),seed,scaling=scaling,time_srs=True,verbose=verbose,thres=0.5)
					avg_score_val=avg_scores_val[2] #hard-coded to optimize F1 score right now, should make this modular
					if max_score<avg_score_val:
						max_score=avg_score_val
						max_parm=parm
						any_change=True
				val_fold+=1
				for i,param in enumerate(parm_ranges):
					if isinstance(param[1],list) and len(param[1])==3:
						param[1][2]=max_parm[i]
				parm_ranges=new_parm_ranges(parm_ranges,graininess=10)
			model=model_init_class(classifier,max_parm)
			x_trainval=x_trval
			y_trainval=y_trval[:,1]
			model.fit(x_trainval,y_trainval)
			self.models[dataset_name]=model
			return model.predict(self.todaysx[dataset_name])

		break_indices=[numb_datapoints-n_folds_test+i for i in range(n_folds_test)]
		performances=[]
		for break_index in break_indices:
			if one_shot:
				#we only test on the very next day!!
				x_trval=x[:break_index]
				x_test=x[break_index:break_index+1]
				y_trval=y[:break_index]
				y_test=y[break_index:break_index+1]
			else:
				#we test on all following days
				x_trval=x[:break_index]
				x_test=x[break_index:]
				y_trval=y[:break_index]
				y_test=y[break_index:]

			self.xdata[dataset_name]=(x_trval,x_test)
			self.ydata[dataset_name]=(y_trval,y_test)
			self._testmode[dataset_name]=True

			max_score=0.
			val_fold=0
			any_change=True
			max_parm=[]
			for parm_range in parm_ranges:
				if isinstance(parm_range[1],list):
					max_parm+=[parm_range[1][2]]
				elif isinstance(parm_range[1],set):
					max_parm+=[list(parm_range[1])[0]]
				else:
					raise KeyError

			while val_fold <parm_search_iter and (any_change or val_fold<5):
				any_change=False
				current_search_space=search_space_builder(parm_ranges,graininess=10)
				for parm in current_search_space:
					if verbose:
						print(parm)
					avg_scores_val=self.kfold_val_class(n_folds_val,dataset_id,classifier,list(parm),seed,scaling=scaling,time_srs=True,verbose=verbose,thres=0.5)
					avg_score_val=avg_scores_val[2] #hard-coded to optimize F1
					if max_score>avg_score_val:
						max_score=avg_score_val
						max_parm=parm
						any_change=True
				val_fold+=1
				for i,param in enumerate(parm_ranges):
					if isinstance(param[1],list) and len(param[1])==3:
						param[1][2]=max_parm[i]
				parm_ranges=new_parm_ranges(parm_ranges,graininess=10)
			
			#performances+=[[min_parm,min_err]]
			#print(performances[-1])
			res=self.kfold_test_class(dataset_id,classifier,max_parm,scaling=scaling,thres=0.5)
			performances+=[res]

		return list(zip(np.mean(performances,axis=0),np.std(performances,axis=0)))

