#am: This is the first module I am adding here, containing NLP preprocessing
import os
import os.path
import datetime
from dateutil.rrule import rrule, DAILY
import pandas as pd
import nltk
import re
from nltk.stem.porter import PorterStemmer
from urllib.parse import urlparse
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

class CorpusGDELT:

	def __init__(self,min_ment=1,datadirec='data/GDELT_1.0/'):
		self.url_corpus=[] #these are lists of raw urls
		self.vect_corpus_bow=pd.DataFrame() #these are the vectorized urls
		self.vect_corpus_tfidf=pd.DataFrame()
		self.vect_bow_uptodate=True
		self.vect_tfidf_uptodate=True
		self.dates=[] #dates added so far
		self.datadirs=[datadirec,] #data directories used so far
		self.currentdir=datadirec #current data directory
		self.header=list(pd.read_csv(self.currentdir+'CSV.header.dailyupdates.txt',delimiter='\t'))
		if min_ment<1:
			self.minimum_ment=1
		else:
			self.minimum_ment=min_ment
		self.re_tokenizer = RegexpTokenizer(r'\w+')
		self.punctuation = re.compile(r'[-.?!,":;()|0-9]')
		self.stop_words = set(stopwords.words('english')+[""])
		self.porter = PorterStemmer()
		self.vowels = list("aeiouy")
		self.consonants = list("bcdfghjklmnpqrstvwxz")
		self.spurious_beginnings = re.compile(r'idind.|idus.|iduk.')
		return

	def change_dir(datadirec):
		self.datadirs+=[datadirec]
		self.currentdir=datadirec
		return
	def dateparser(self,date1,date2):
		#this will take in two dates in 'yyyymmdd' format (e.g. '20140110') and spit out a range
		date1=datetime.date(int(date1[:4]),int(date1[4:6]),int(date1[6:]))
		date2=datetime.date(int(date2[:4]),int(date2[4:6]),int(date2[6:]))
		new_dates=[]
		for date in rrule(DAILY, dtstart=date1, until=date2):
			datestr=date.strftime("%Y%m%d")
			if datestr not in self.dates:
				new_dates+=[datestr]
		return new_dates

	def reader(self,file_path):
		if not os.path.isfile(file_path):
			os.system('wget http://data.gdeltproject.org/events/'+date+'.export.CSV.zip')
			os.system('unzip '+date+'.export.CSV.zip')
			os.system('mv '+date+'.export.CSV '+self.currentdir)
			os.system('rm '+date+'.export.CSV.zip')
		df=pd.read_csv(file_path,delimiter='\t')
		df.columns=self.header
		return df


	def load_urls(self,date1,date2,save=False):
		#here I don't remember what I was writing to remember to do...
		new_dates=self.dateparser(date1,date2)
		for date in new_dates:
			file_path=self.currentdir+date+'.export.CSV'
			df=self.reader(file_path)
			url_doc=[]
			if self.minimum_ment<2:
				for i in range(len(df)):
					url_doc+=[[df['NumMentions'][i],df.iloc[i,-1]]]
			else:
				for i in range(len(df)):
					numb_ment=df['NumMentions'][i]
					if numb_ment>=self.minimum_ment:
						url_doc+=[[numb_ment,df.iloc[i,-1]]]
			self.url_corpus+=[url_doc]
		if save:
			print("Sorry, I haven't implemented a saving feature yet")
		self.dates+=new_dates
		if len(new_dates)>0:
			self.vect_bow_uptodate=False
			self.vect_tfidf_uptodate=False
		return
	def url_tokenizer(self,url):
		c,d,e=[],[],[]
		if url!='BBC Monitoring':
			a=urlparse(url)[2].split('.')[0].split('/')[-1]
			b = self.re_tokenizer.tokenize(a.lower())
			for word in b:
				c+=[self.punctuation.sub("", word)]
			for word in c:
				if word not in self.stop_words:
					d+=[word]
			if len(d)<=1:
				return []
			for word in d:
				stemtemp=self.porter.stem(word)
				length=len(stemtemp)
				unique=len(set(stemtemp))
				num_vow=sum(stemtemp.count(c) for c in self.vowels)
				num_cons=sum(stemtemp.count(c) for c in self.consonants)
				if length<15 and (num_cons-num_vow)<7 and unique>1 and num_vow>0 and (length-unique)<5 and not self.spurious_beginnings.match(stemtemp) and '_' not in stemtemp:
					e+=[stemtemp]
		return e

	def wrapper_tokenizer(self,url_doc):
		wordlist=[]
		for url in url_doc:
			for mentions in range(url[0]):
				wordlist+=self.url_tokenizer(url[1])
		return wordlist
	def gdelt_preprocess(self,tfidf=False):
		if tfidf:
			vectorizer = TfidfVectorizer(min_df=1,tokenizer=self.wrapper_tokenizer,lowercase=False)
		else:
			vectorizer = CountVectorizer(min_df=1,tokenizer=self.wrapper_tokenizer,lowercase=False)
		X = vectorizer.fit_transform(self.url_corpus).toarray()
		dictionary={col:X[:,i] for i,col in enumerate(vectorizer.get_feature_names())}
		if tfidf:
			self.vect_corpus_tfidf=pd.DataFrame(dictionary)
			self.vect_tfidf_uptodate=True
		else:
			self.vect_corpus_bow=pd.DataFrame(dictionary)
			self.vect_bow_uptodate=True
		return 
