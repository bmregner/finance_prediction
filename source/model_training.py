"""
This module creates datasets from input dataframes of (nlp) feature vectors and trains and validates
a variety of simple scikitlearn models on them. This module is in constant process of generalization.
"""
import re
import numpy as np
import csv
import datetime
from sklearn.model_selection import train_test_split, KFold
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



class StockPrediction:
	"""
	This class creates datasets from input dataframes of (nlp) feature vectors and trains and
	 validates a variety of simple scikitlearn models on them. For generality and ease of model 
	 exploration, it stores all the processed data (x/ydata) as dictionaries of numpy arrays, ready 
	 to be fed to ML algorithms, also predictions (yhat_*) keeping track of test/validation stage 
	 (testmode) and in the future of model parameters and settings.
	"""

	def __init__(self,dataframes_list=[[],[]],gdeltcorp_list=False,sp500t=True):
		self.dataset_names=dataframes_list[0]
		self.dataset_list=dataframes_list[1]
		if gdeltcorp_list!=False:
			for gdeltcorp in gdeltcorp_list:
				var_name = [k for k,var in locals().items() if var is gdeltcorp][0]
				self.dataset_names+=[var_name+'_bow',var_name+'_tfidf']
				self.dataset_list+=[gdeltcorp.vect_corpus_bow,]
				self.dataset_list+=[gdeltcorp.vect_corpus_tfidf,]
		#list of used models, not yet really used
		self.models=[]
		#have we already split the data with key 'something' into training+validation and testing?
		self.testmode={}
		#nlp+time series features for dataset of key 'something'
		self.xdata={}
		#target data for regression, incremental regression, and bullish/bearish classification
		self.ydata={}
		self.sp500={}
		self.yhat_reg={}
		self.yhat_reg_diff={}
		self.yhat_class={}
		if sp500t:
			self.load_sp500()
		return

	def import_nlp_data(self,dataframes_list=[[],[]],gdeltcorp_list=False):
		"""
		This method essentially works in the same exact way as the initializer up above, 
		simply adding to the data already there. Please see above for comments.
		"""
		self.dataset_names+=dataframes_list[0]
		self.dataset_list+=dataframes_list[1]
		if gdeltcorp_list !=False:
			for gdeltcorp in gdeltcorp_list:
				var_name = [k for k,var in locals().items() if var is gdeltcorp][0]
				self.dataset_names+=[var_name+'_bow',var_name+'_tfidf']
				self.dataset_list+=[gdeltcorp.vect_corpus_bow,]
				self.dataset_list+=[gdeltcorp.vect_corpus_tfidf,]
		return

	def add_model(self,model_list):
		"""
		Probably deprecated, it adds to the used models but I think I won't be adding models like 
		this, only when fitting them.
		"""
		self.models+=model_list
		return

	def load_sp500(self):
		"""
		This method loads S&P500 index data, probably it could be part of the initializer but for 
		generality it's separate. One could envision a future where other loaders load different 
		time series to predict.
		"""
		with open('data/SP500am.csv','r') as mycsvfile:
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

	def prepare_data(self,dataset_id):
		"""
		This method simply read a certain dataframe labeled by a certain name or ordinal number (dataset_id)
		together with the targets y and converts this information into numpy arrays stored in dictionaries
		"""
		if isinstance(dataset_id,int):
			index_df=dataset_id
			dataset_name=self.dataset_names[dataset_id]
		else:
			dataset_name=dataset_id
			index_df=self.dataset_names.index(dataset_name)
		dataframe=self.dataset_list[index_df]
		days=list(dataframe.index.values)
		self.xdata[dataset_name]=self.prepare_x(dataframe,days)
		self.ydata[dataset_name]=self.prepare_y(days)
		self.testmode[dataset_name]=False
		return

	def prepare_x(self,dataframe,days):
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


	def prepare_y(self,days):
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
				y_arr+=[[float(sp[tomorrow][-2]),(np.sign(difference)+1)/2.,difference]]
		return np.array(y_arr)


	def trval_test_split(self,dataset_id,test_frac):
		"""
		This method randomly splits the dataset labelled by 'dataset_id' into training+validation and
		test datasets in a ratio of test_frac
		"""
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

		#using scikit learn splitting function
		x_trval, x_test, y_trval, y_test = train_test_split(x,y,test_size=test_frac)

		#storing the data into the same dictionary entries they came from, notice that this creates 
		#a different format, because now each dict value is no longer an array but a tuple of two arrays:
		#(train+val,test). I keep track of this with the 'testmode' attribute
		self.xdata[dataset_name]=(x_trval,x_test)
		self.ydata[dataset_name]=(y_trval,y_test)
		self.testmode[dataset_name]=True
		return

	def kfold_val_reg(self,n_folds_val,dataset_id,regressor,parm,seed,differential=False,scaling=False):
		"""
		This method performs a k-fold validation of a regression model on a certain dataset labelled by 
		'dataset_id', with a certain regressor, with certain parameters 'parm', distinguishing between a
		full tomorrow's value prediction and a tomorrow's increment prediction (argument differential).
		Scaling of (all) the features is also available but not recommended for now b.c. of poor results.
		For reproducibilty, I also have a random k-folding seed.
		Example input: kfold_val_reg(10,'apriltodectfidf','lasso',1.3,10,scaling=True)
		"""
		regressor=modeldict[regressor]
		if isinstance(dataset_id,int):
			dataset_name=self.dataset_names[dataset_id]
		else:
			dataset_name=dataset_id
		x_trainval=self.xdata[dataset_name][0]
		if differential:
			y_trainval=self.ydata[dataset_name][0][:,2]
		else:
			y_trainval=self.ydata[dataset_name][0][:,0]
		#the seed comes in here
		kf_val = KFold(n_splits=n_folds_val,shuffle=True,random_state=seed)
		avg_rms_mod_val=0
		avg_rms_mod_train=0
		#splitting validation from training, scaling is optional
		for train_index, val_index in kf_val.split(x_trainval):
			if scaling:
				x_train, x_val = scale(x_trainval[train_index]), scale(x_trainval[val_index])
				y_train, y_val = y_trainval[train_index], y_trainval[val_index]
			else:
				x_train, x_val = x_trainval[train_index], x_trainval[val_index]
				y_train, y_val = y_trainval[train_index], y_trainval[val_index]
			#parsing parameters and initializing model, given the chosen regressor
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
				print('houston, we have an unknown model problem')
				return
			model.fit(x_train,y_train)
			#storing the train/validation rmse performances
			avg_rms_mod_val+=np.sqrt((sum((model.predict(x_val)-y_val)**2)/len(y_val)))
			avg_rms_mod_train+=np.sqrt((sum((model.predict(x_train)-y_train)**2)/len(y_train)))
		#averaging over the several folds
		avg_rms_mod_val=avg_rms_mod_val/n_folds_val
		avg_rms_mod_train=avg_rms_mod_train/n_folds_val
		#printing the average train/validation rmse performances
		print('avg_train_rmse:',avg_rms_mod_train,'avg_validation_rmse:',avg_rms_mod_val)
		return

	def kfold_test_reg(self,dataset_id,regressor,parm,differential=False,scaling=False):
		"""
		Once a model is finally picked, given its performance on the validation sets (model tuning),
		we should see how it performs on the never-seen-before test set. This is what the method does.
		"""
		regressor=modeldict[regressor]
		if isinstance(dataset_id,int):
			dataset_name=self.dataset_names[dataset_id]
		else:
			dataset_name=dataset_id


		if scaling:
			scale(x_trainval=self.xdata[dataset_name][0])
			scale(x_test=self.xdata[dataset_name][1])
		else:
			x_trainval=self.xdata[dataset_name][0]
			x_test=self.xdata[dataset_name][1]

		#for regression, this is my benchmark model, that next day's prediction is equal to today's closing
		y_flat=x_test[:,-1]


		if differential:
			y_trainval=self.ydata[dataset_name][0][:,2]
			y_test=self.ydata[dataset_name][1][:,2]
		else:
			y_trainval=self.ydata[dataset_name][0][:,0]
			y_test=self.ydata[dataset_name][1][:,0]

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
			print('houston, we have an unknown model problem')        
			return
		model.fit(x_trainval,y_trainval)

		rms_mod_test=np.sqrt((sum((model.predict(x_test)-y_test)**2)/len(y_test)))

		# if we are perfoming tomorrow-today prediction, then my benchmark model is a prediction of 0...
		if differential:
			rms_rand_test=np.sqrt((sum((y_test)**2)/len(y_test)))
			self.yhat_reg_diff[dataset_name]=model.predict(x_test)		
		# ...otherwise a flat prediction	
		else:
			rms_rand_test=np.sqrt((sum((y_flat-y_test)**2)/len(y_test)))
			self.yhat_reg[dataset_name]=model.predict(x_test)			

		#aaand these are the rmse performances of the model vs the benchmark
		print('model_test_rmse:',rms_mod_test,'flat_test_rmse:',rms_rand_test)

		#here I return the model parameters if the option is implemented in scikit learn
		if coeff:
			return model.coef_
		return

	def kfold_val_class(self,n_folds_val,dataset_id,classifier,parm,seed,thres):
		"""
		This method performs a k-fold validation of a classification model on a certain dataset labelled by 
		'dataset_id', with a certain classifier, with certain parameters 'parm'. A list of classification 
		thresholds has to be provided as well to feed the outside function 'scores' (see way above), but 
		in practice for now leave it to [0.5].
		For reproducibilty, I also have a random k-folding seed.
		Example input: kfold_val_reg(10,'apriltodectfidf','logreg',1.3,10,[0.5])
		Please see the regression methods above for comments.
		"""
		classifier=modeldict[classifier]
		if isinstance(dataset_id,int):
			dataset_name=self.dataset_names[dataset_id]
		else:
			dataset_name=dataset_id
		x_trainval=self.xdata[dataset_name][0]
		y_trainval=self.ydata[dataset_name][0][:,1]

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
				print('houston, we have an unknown model problem')
				return
			model.fit(x_train,y_train)
			#the difference with the regressor is that the performance is evaluated with prec, reg and f1
			avg_scores_train+=np.array(scores(y_train,model.predict(x_train),[thres])[:3])
			avg_scores_val+=np.array(scores(y_val,model.predict(x_val),[thres])[:3])

		avg_scores_train=avg_scores_train/n_folds_val    
		avg_scores_val=avg_scores_val/n_folds_val
		print('avg_train_rec,prec,F1:',list(avg_scores_train),'avg_validation_rec,prec,F1:',list(avg_scores_val))
		return

	def kfold_test_class(self,dataset_id,classifier,parm,thres):
		"""
		Once a model is finally picked, given its performance on the validation sets (model tuning),
		we should see how it performs on the never-seen-before test set. This is what the method does.
		Please see the regression methods above for comments.
		"""
		classifier=modeldict[classifier]
		if isinstance(dataset_id,int):
			dataset_name=self.dataset_names[dataset_id]
		else:
			dataset_name=dataset_id
		x_trainval=self.xdata[dataset_name][0]
		y_trainval=self.ydata[dataset_name][0][:,1]
		x_test=self.xdata[dataset_name][1]
		y_test=self.ydata[dataset_name][1][:,1]

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
			print('houston, we have an unknown model problem')
			return

		model.fit(x_trainval,y_trainval)

		self.yhat_class[dataset_name]=model.predict(x_test)
		scores_test=np.array(scores(y_test,self.yhat_class[dataset_name],[thres])[:3])
		print('test_rec,prec,F1:',list(scores_test))   
		
		if coeff:
			return model.coef_
		return