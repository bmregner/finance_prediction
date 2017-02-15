"""
This module contains NLP preprocessing. It has two classes, each processing data from a different source: 
GDELT and Google News respectively.
"""
import os
import os.path
import datetime
import json
from dateutil.rrule import rrule, DAILY
import numpy as np
import pandas as pd
import nltk
import re
from nltk.stem.porter import PorterStemmer
from urllib.parse import urlparse
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class CorpusGDELT:
	"""
	This class contains the NLP preprocessing for the GDELT corpus. It allows to download, load and 
	turn into vectorized feature dataframes, any subset of the corpus of urls found in the GDELT 
	project files.
	"""

	def __init__(self,min_ment=1,datadirec='../data/GDELT_1.0/'):
		"""
		The initializer mainly builds empty attributes, apart from the mentions cutoff which might be 
		in the hundreds for concreteness (but it's defaulted to 1, that is no cutoff at all), and apart
		from the working directory, which however defaults to 'data/GDELT_1.0/'. For the moment there is 
		no option to deal with a non existent folder so the user has to create it manually but I'll soon 
		handle this.
		"""
		self.url_corpus=[] #these are lists of raw urls
		self.vect_corpus_bow=pd.DataFrame() #these are the vectorized urls
		self.vect_corpus_tfidf=pd.DataFrame()
		self.word2vec_corpus=pd.DataFrame()
		#model memorized
		self.w2vec_model=None
		#the following three attributes tell you if we have reprocessed the url corpus after loading new urls
		self.vect_bow_uptodate=(True,'all')
		self.vect_tfidf_uptodate=(True,'all')
		self.word2vec_uptodate=(True,'all',0)
		self.dates=[] #dates added so far
		self.datadirs=[datadirec,] #data directories used so far, not sure if useful
		self.currentdir=datadirec #current data directory
		#this reads the columns header for the GDELT files, again no error handling so far
		self.header=list(pd.read_csv(self.currentdir+'CSV.header.dailyupdates.txt',delimiter='\t'))
		if min_ment<1:
			self.minimum_ment=1
		else:
			self.minimum_ment=min_ment
		#the following attributes all store NLP processing choices, which for now are not left free for the
		# user to set. Used in the url_tokenizer method
		self.re_tokenizer = RegexpTokenizer(r'\w+')
		self.punctuation = re.compile(r'[-.?!,":;()|0-9]')
		self.stop_words = set(stopwords.words('english')+[""])
		self.porter = PorterStemmer()
		self.vowels = list("aeiouy")
		self.consonants = list("bcdfghjklmnpqrstvwxz")
		self.spurious_beginnings = re.compile(r'idind.|idus.|iduk.')
		self._dtypes={8:str,9:str,10:str,11:str,12:str,13:str,14:str,18:str,19:str,20:str,21:str,23:str,24:str,26:str,27:str,28:str,40:str,47:str,54:str}
		return

	def change_dir(self,datadirec):
		"""
		This simply changes the working directory
		and adds the new one to the list of all the directories used
		"""
		self.datadirs+=[datadirec]
		self.currentdir=datadirec
		return

	def new_dateparser(self,date1,date2):
		"""
		This takes in two dates in 'yyyymmdd' format (e.g. '20140110') and spits out 
		a range of valid dates that haven't been loaded yet, in the same format
		"""
		date1=datetime.date(int(date1[:4]),int(date1[4:6]),int(date1[6:]))
		date2=datetime.date(int(date2[:4]),int(date2[4:6]),int(date2[6:]))
		new_dates=[]
		for date in rrule(DAILY, dtstart=date1, until=date2):
			datestr=date.strftime("%Y%m%d")
			if datestr not in self.dates and int(datestr[:6])>201303:#earliest available date is 20130401
				new_dates+=[datestr]
		return new_dates

	def old_dateparser(self,date1,date2):
		"""
		This takes in two dates in 'yyyymmdd' format (e.g. '20140110') and spits out 
		a range of valid dates that have already been loaded, in the same format
		"""
		date1=datetime.date(int(date1[:4]),int(date1[4:6]),int(date1[6:]))
		date2=datetime.date(int(date2[:4]),int(date2[4:6]),int(date2[6:]))
		old_dates=[]
		for date in rrule(DAILY, dtstart=date1, until=date2):
			datestr=date.strftime("%Y%m%d")
			if datestr in self.dates and int(datestr[:6])>201303:#earliest available date is 20130401
				old_dates+=[datestr]
		return old_dates

	def _reader(self,file_path,date):
		"""
		This method reads a dataframe out of a csv file that represents one day of gdelt news. If the file 
		doesn't already exist in the curent working directory, it'll download it and process it
		"""
		if not os.path.isfile(file_path):
			print("\r",'file for',date,'not already there, so downloading it from http://data.gdeltproject.org/events/ ...',end="")
			os.system('wget -nv http://data.gdeltproject.org/events/'+date+'.export.CSV.zip')
			os.system('unzip '+date+'.export.CSV.zip')
			os.system('mv '+date+'.export.CSV '+self.currentdir)
			os.system('rm '+date+'.export.CSV.zip')
		df=pd.read_csv(file_path,delimiter='\t',dtype=self._dtypes)
		df.columns=self.header
		return df

	def load_urls(self,date1,date2,save=False):
		"""
		This method will read in the relevant dataframe for each date in the range date1-date2, 
		after which it'll extract the urls and the number of mentions of each event and store them 
		into the corpus class attribute as a list of documents, where each document is a 
		list of several [numb_mentions,url].
		"""
		new_dates=self.new_dateparser(date1,date2)
		if len(new_dates)>0:
			#loading new urls! So the processed data isn't up-to-date any more
			self.vect_bow_uptodate=(False,'all',)
			self.vect_tfidf_uptodate=(False,'all',)
			self.word2vec_uptodate=(False,'all',0)
		for date in new_dates:
			print("\r",'loading news for',date,'...',end="")
			file_path=self.currentdir+date+'.export.CSV'
			df=self._reader(file_path,date)
			url_doc=[]
			#here is where I apply the mentions cutoff, of course only if the cutoff is larger than 1
			if self.minimum_ment<2:
				for i in range(len(df)):
					url_doc+=[[df['NumMentions'][i],df.iloc[i,-1]]]
			else:
				for i in range(len(df)):
					numb_ment=df['NumMentions'][i]
					#mentions cutoff right here
					if numb_ment>=self.minimum_ment:
						url_doc+=[[numb_ment,df.iloc[i,-1]]]
			self.url_corpus+=[url_doc]
			self.dates+=[date]
		if save:
			print("Sorry, I haven't implemented a saving feature yet")
		print("\r",'Done!',end="")
		return

	def url_tokenizer(self,url):
		"""
		This method is my customized url tokenizer, it takes in one url and returns a list of
		relevants tokens, after stemming, filtering out stopwords and other spurious tokens.
		"""
		filter3,filter4,filter5=[],[],[] #these will be sequential lists of words, being filtered more 
		#and more of their unwanted words
		if url!='BBC Monitoring':
			#print(urlparse(url)[2].split('.')[0].split('/'))
			#filter1=urlparse(url)[2].split('.')[0].split('/')[-1]
			filter1=max(urlparse(url)[2].split('.')[0].split('/'),key=len)
			filter2 = self.re_tokenizer.tokenize(filter1.lower())
			for word in filter2:
				filter3+=[self.punctuation.sub("", word)]
			for word in filter3:
				if word not in self.stop_words:
					filter4+=[word]
			if len(filter4)<=1:
				return []
			for word in filter4:
				#stemmer!
				stemtemp=self.porter.stem(word)
				length=len(stemtemp)
				unique=len(set(stemtemp))
				num_vow=sum(stemtemp.count(c) for c in self.vowels)
				num_cons=sum(stemtemp.count(c) for c in self.consonants)
				#several heuristic/exploration-based cutoffs are here applied to filter out meaningless words
				if length<15 and (num_cons-num_vow)<7 and unique>1 and num_vow>0 and (length-unique)<5 and not self.spurious_beginnings.match(stemtemp) and '_' not in stemtemp:
					filter5+=[stemtemp]
		return filter5

	def wrapper_tokenizer(self,url_doc):
		"""
		This method is simply a wrapper for my tokenizer to make it consistent with the callable
		tokenizer that scikitlearn vectorizers need to be passed. It takes in a document 
		(= list of [numb_mentions,url] pairs) and returns a large list of tokens
		"""
		wordlist=[]
		for url in url_doc:
			for mentions in range(url[0]):
				wordlist+=self.url_tokenizer(url[1])
		return wordlist

	def gdelt_preprocess(self,vectrz='word2vec',size_w2v=20,daterange='all'):
		"""
		This is the vectorizer! It can be called with bag of words, tfidf or word2vec approach. 
		It populates the "corpus" fields of the class with dataframes processing from the url corpus.
		The input daterange can be 'all' or 
		"""
		lildict={'word2vec':'word2vec','bow':'bag-of-words','tfidf':'Tf-Idf'}
		print('Using',lildict[vectrz],'vectorization procedure')
		if (vectrz=='word2vec' and self.word2vec_uptodate==(True,daterange,size_w2v) ) or (vectrz=='bow' and self.vect_bow_uptodate==(True,daterange,)) or (vectrz=='tfidf' and self.vect_tfidf_uptodate==(True,daterange,)):
			print('Nothing to be done, dataframes are up to date')
			return
		
		if daterange=='all':
			url_corpus=self.url_corpus
			old_dates=self.dates
		elif len(daterange)==2:
			old_dates=self.old_dateparser(daterange[0],daterange[1])
			url_corpus=[datedoc[1] for i,date_doc in enumerate(zip(self.dates,self.url_corpus)) if date_doc[0] in old_dates]
		else:
			print("optional parameter 'daterange' was not entered according to the correct format: e.g.['20140520',20140615]")			

		#word2vec!
		if vectrz=='word2vec':
			corpus=[sum([self.url_tokenizer(url[1]) for url in news_day],[]) for news_day in url_corpus]
			model = gensim.models.Word2Vec(corpus, size=size_w2v, min_count=1)
			self.w2vec_model=model
			dictionary={}
			for col in range(size_w2v):
				dictionary['w2v_'+str(col+1)]=[sum([model[word][col] for word in news_day]) for news_day in corpus]
			#vectorized=np.array()
			#print(vectorized)
			dictionary['news_date']=old_dates
			dataf=pd.DataFrame(dictionary).set_index('news_date')
		#defining the vectorizer and passing it my callable custom tokenizer
		else:
			if vectrz=='tfidf':
				vectorizer = TfidfVectorizer(min_df=1,tokenizer=self.wrapper_tokenizer,lowercase=False)
			else:
				vectorizer = CountVectorizer(min_df=1,tokenizer=self.wrapper_tokenizer,lowercase=False)
			#in the follwing vectorizing and writing into a dataframe
			X = vectorizer.fit_transform(url_corpus).toarray()
			dictionary={col:X[:,i] for i,col in enumerate(vectorizer.get_feature_names())}
			dictionary['news_date']=old_dates
			dataf=pd.DataFrame(dictionary).set_index('news_date')

		#deleting empty days


		droppers=[row_ind for row_ind in range(len(dataf)) if sum(abs(dataf.iloc[row_ind]))==0.0]
		print(droppers)
		#temp_len=len(dataf)
		#for row_ind in range(temp_len):
		#	rev_ind=temp_len-row_ind-1
		#	if sum(abs(dataf.iloc[rev_ind]))==0.0:
		dataf.drop(droppers)
		#populating the dataset attributes and setting these attributes to True because now the newly 
		#added urls have been processed and the data is up-to-date
		if vectrz=='tfidf':
			self.vect_corpus_tfidf=dataf
			self.vect_tfidf_uptodate=(True,daterange)
		elif vectrz=='bow':
			self.vect_corpus_bow=dataf
			self.vect_bow_uptodate=(True,daterange)
		elif vectrz=='word2vec':
			self.word2vec_corpus=dataf
			self.word2vec_uptodate=(True,daterange,size_w2v)
		return


class CorpusGoogleNews:
	"""
	This class contains the NLP preprocessing from the google news corpus
	"""

	def __init__(self,path='../data/'):
		self.datadirectory=path
		self.raw_articles={}
		return

	
	def _read_Google_articles(self,articlelist, path):
		""" 
		Converts Google News JSON file into a data frame. Takes in
		a .json file and returns a dataframe using the json's dictionary-like
		structure 
		"""
		
		with open(path + articlelist[0],'r') as first:
			firstdict = json.load(first)
			combined_df = pd.DataFrame.from_dict(firstdict, orient = 'index')
			combined_df = combined_df.T
		
		for article in articlelist:
			with open(path + article, 'r') as fin:
				mydict = json.load(fin)
			current_df = pd.DataFrame.from_dict(mydict, orient = 'index')
			current_df = current_df.T
			combined_df = combined_df.append(current_df, ignore_index=True)
		
		# USE CONCAT WITH .APPEND DOESN'T WORK!!!
		#     final_df = pd.concat([combined_df, current_df])
			
		return combined_df

	def data_directory_crawl(self, ticker,verbose=1):
		"""
		Crawls through a given parent directory to create a dataframe of articles and their body for the given company ticker
		"""
		path=self.datadirectory
		mypath = path + ticker + '/'
		company_articles_combined_days=pd.DataFrame()

		for directory in os.listdir(mypath):
			if verbose:
				print(directory)
			f = []
			d = []
			for (dirpath, dirnames, filenames) in os.walk(mypath + directory):
				f.extend(filenames)
				d.extend(dirnames)

			company_articles_combined_days = company_articles_combined_days.append(self._read_Google_articles(f, mypath + directory + '/'))
			if directory not in self.raw_articles.keys():
				self.raw_articles[directory]=company_articles_combined_days

		return



