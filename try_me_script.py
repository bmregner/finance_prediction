import sys
import os
import datetime
sourcedir=os.getcwd()+"/source"
datadir=os.getcwd()+"/data/GDELT_1.0/"
datadir_sp500=os.getcwd()+"/data/"
if sourcedir not in sys.path:
    sys.path.append(sourcedir)
import nlp_preprocessing as nlpp
import model_training as mdlt

numb_val_folds=20
numb_past_days=30

tot_days=3*(numb_val_folds+numb_past_days)//2+1

if datetime.datetime.utcnow().hour < 11:
	todays_date=datetime.date.today()-datetime.timedelta(days=1)
else:
	todays_date=datetime.date.today()
yesterday=todays_date-datetime.timedelta(days=1)
hundred_days_ago=yesterday-datetime.timedelta(days=tot_days)

todays_date_str=todays_date.strftime("%Y%m%d")
yesterday_str=yesterday.strftime("%Y%m%d")
hundred_days_ago_str=hundred_days_ago.strftime("%Y%m%d")

print("Today's date (or the latest date available) is",todays_date)
print("Let's start by loading the data for the previous",tot_days,"days, that's how much I'll need. Ok?")

gdelt=nlpp.CorpusGDELT(datadirec=datadir,min_ment=500)
gdelt.load_urls(hundred_days_ago_str,yesterday_str)

print("Now, we're gonna process the urls using the magic of word2vec!")

gdelt.gdelt_preprocess(vectrz='word2vec',size_w2v=8)

print("Now it's time to load the stock market data.")

trainer=mdlt.StockPrediction([['word2vec'],[gdelt.word2vec_corpus],[gdelt.w2vec_model]],update=True,datadir=datadir_sp500)

print("And now we're gonna make our prediction! Abracadabra...")

#out=trainer.auto_ts_val_test_reg('word2vec','lasso',[['alpha',[0.2,5.0,1.2]]],parm_search_iter=15,n_folds_val=10,n_folds_test=10,past_depth=10,scaling=False,differential=False,verbose=False,notest=True)
out=trainer.auto_ts_val_test_reg('word2vec','knnreg',[['numb_nn',{3,4,5,6}]],parm_search_iter=10,n_folds_val=numb_val_folds,n_folds_test=1,past_depth=numb_past_days,scaling=True,differential=True,verbose=False,notest=True,eqdiff=False)

if out is not None:
	print('Predicted s&p500 closing for today,',todays_date,':',out[0],'. Enjoy your millions.')

	#res=trainer.models['word2vec'].coef_[:32]
	#w2vec_model=trainer.w2v_models['word2vec']
	#print(w2vec_model[res])
	#print(w2vec_model[-res])

print('I can also just predict if the index will go up or down...')
out=trainer.auto_ts_val_test_class('word2vec','knnclass',[['numb_nn',{3,4,5,6}]],parm_search_iter=10,n_folds_val=numb_val_folds,n_folds_test=1,past_depth=numb_past_days,scaling=True,verbose=False,notest=True)
if out is not None:
	mydict={1:'higher',0:'lower'}
	print('I think that the s&p500 closing for today will be '+mydict[int(out[0])]+' than yesterday. Will I be right??')


print('Adieu')


