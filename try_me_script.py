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

if datetime.datetime.utcnow().hour < 11:
	todays_date=datetime.date.today()-datetime.timedelta(days=1)
else:
	todays_date=datetime.date.today()
yesterday=todays_date-datetime.timedelta(days=1)
hundred_days_ago=yesterday-datetime.timedelta(days=50)

todays_date_str=todays_date.strftime("%Y%m%d")
yesterday_str=yesterday.strftime("%Y%m%d")
hundred_days_ago_str=hundred_days_ago.strftime("%Y%m%d")

print("Today's date is",todays_date,". Let's start by loading the data for the last 100 days, ok?")

gdelt=nlpp.CorpusGDELT(datadirec=datadir,min_ment=300)
gdelt.load_urls(hundred_days_ago_str,yesterday_str)

print("Now, we're gonna process the urls using the magic of word2vec")

gdelt.gdelt_preprocess(vectrz='word2vec',size_w2v=32)

print("Now it's time to load the stock market data!")

trainer=mdlt.StockPrediction([['word2vec'],[gdelt.word2vec_corpus],[gdelt.w2vec_model]],update=True,datadir=datadir_sp500)

print("And now we're gonna make our prediction!")

out=trainer.auto_ts_val_test_reg('word2vec',100,'lasso',[['alpha',[0.2,5.0,1.2]]],parm_search_iter=15,n_folds_val=10,n_folds_test=10,max_test_frac=0.2,scaling=False,differential=False,one_shot=True,verbose=True,notest=True)

print('predicted s&p500 closing for today:',out[0])

res=trainer.models['word2vec'].coef_[:32]
w2vec_model=trainer.w2v_models['word2vec']
print(w2vec_model[res])
print(w2vec_model[-res])



