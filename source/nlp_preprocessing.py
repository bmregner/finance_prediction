"""
This module contains NLP preprocessing. It has two classes, each processing data from a different source: 
GDELT and Google News respectively.
"""
import os
import os.path
import datetime
import json
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
		#the following two attributes tell you if we have reprocessed the url corpus after loading new urls
		#into the class
		self.vect_bow_uptodate=True
		self.vect_tfidf_uptodate=True
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
		self._dtypes={8:str,9:str,10:str,11:str,12:str,13:str,14:str,18:str,19:str,20:str,21:str,23:str,24:str,26:str,27:str,28:str}
		return

	def change_dir(self,datadirec):
		"""
		This simply changes the working directory
		and adds the new one to the list of all the directories used
		"""
		self.datadirs+=[datadirec]
		self.currentdir=datadirec
		return

	def dateparser(self,date1,date2):
		"""
		This takes in two dates in 'yyyymmdd' format (e.g. '20140110') and spits out 
		a range of valid dates in the same format
		"""
		date1=datetime.date(int(date1[:4]),int(date1[4:6]),int(date1[6:]))
		date2=datetime.date(int(date2[:4]),int(date2[4:6]),int(date2[6:]))
		new_dates=[]
		for date in rrule(DAILY, dtstart=date1, until=date2):
			datestr=date.strftime("%Y%m%d")
			if datestr not in self.dates and int(datestr[:6])>201303:#earliest available date is 20130401
				new_dates+=[datestr]
		return new_dates

	def _reader(self,file_path,date):
		"""
		This method reads a dataframe out of a csv file that represents one day of gdelt news. If the file 
		doesn't already exist in the curent working directory, it'll download it and process it
		"""
		if not os.path.isfile(file_path):
			print('file not already there, so downloading it from http://data.gdeltproject.org/events/')
			os.system('wget http://data.gdeltproject.org/events/'+date+'.export.CSV.zip')
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
		new_dates=self.dateparser(date1,date2)
		if len(new_dates)>0:
			#loading new urls! So the processed data isn't up-to-date any more
			self.vect_bow_uptodate=False
			self.vect_tfidf_uptodate=False
		for date in new_dates:
			print('loading news for '+date)
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
		return

	def url_tokenizer(self,url):
		"""
		This method is my customized url tokenizer, it takes in one url and returns a list of
		relevants tokens, after stemming, filtering out stopwords and other spurious tokens.
		"""
		filter3,filter4,filter5=[],[],[] #these are sequential lists of words, being filtered more 
		#and more of their unwanted words
		if url!='BBC Monitoring':
			filter1=urlparse(url)[2].split('.')[0].split('/')[-1]
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

	def gdelt_preprocess(self,tfidf=False):
		"""
		This is the vectorizer! It can be called with or without tfidf approach. It populates
		the "corpus" fields of the class with dataframes processing from the url corpora
		"""
		if (tfidf and self.vect_tfidf_uptodate) or (not tfidf and self.vect_bow_uptodate):
			print('Nothing to be done, dataframes are up to date')
			return
		#defining the vectorizer and passing it my callable custom tokenizer
		if tfidf:
			vectorizer = TfidfVectorizer(min_df=1,tokenizer=self.wrapper_tokenizer,lowercase=False)
		else:
			vectorizer = CountVectorizer(min_df=1,tokenizer=self.wrapper_tokenizer,lowercase=False)
		#in the follwing vectorizing and writing into a dataframe
		X = vectorizer.fit_transform(self.url_corpus).toarray()
		dictionary={col:X[:,i] for i,col in enumerate(vectorizer.get_feature_names())}
		dictionary['news_date']=self.dates
		dataf=pd.DataFrame(dictionary).set_index('news_date')
		#populating the dataset attributes and setting these attributes to True because now the newly 
		#added urls have been processed and the data is up-to-date
		if tfidf:
			self.vect_corpus_tfidf=dataf
			self.vect_tfidf_uptodate=True
		else:
			self.vect_corpus_bow=dataf
			self.vect_bow_uptodate=True

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



