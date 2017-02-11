"""
This is a script for testing functions and methods in the nlp preprocessing module
"""
import os
import sys
sourcedir=os.path.split(os.getcwd())[0]+"/maqro/source"
datadir=os.path.split(os.getcwd())[0]+"/maqro/data/GDELT_1.0/"
if sourcedir not in sys.path:
    sys.path.append(sourcedir)
#print(sourcedir)
#print(sys.path)
import nlp_preprocessing as nlpp
import pytest


test_class_init_data=[([-2],'minimum_ment',1),([2],'minimum_ment',2),([2],'dates',[]),([2],'url_corpus',[])]
@pytest.mark.parametrize('parameters,attribute,expected',test_class_init_data)
def test_class_init(parameters,attribute,expected):
	"""
	Very beginning of testing of the initializer, bear with me while I expand this module
	"""
	if len(parameters)==1:
		cl_inst=nlpp.CorpusGDELT(min_ment=parameters[0],datadirec=datadir)
	else:
		cl_inst=nlpp.CorpusGDELT(min_ment=parameters[0],datadirec=parameters[1])
	assert getattr(cl_inst,attribute)==expected
	return

stinit=[300,'20130927','20131001']
nonstinit=[300,'20130301','20130401']
test_load_urls_data=[(stinit,'dates',5),(nonstinit,'dates',1),(stinit,'url_corpus1',None),(stinit,'url_corpus2',None),(stinit,'url_corpus3',None),(stinit,'url_corpus4',7)]
@pytest.mark.parametrize('parameters,attribute,expected',test_load_urls_data)
def test_load_urls(parameters,attribute,expected):
	"""
	Testing loading urls
	"""
	cl_inst=nlpp.CorpusGDELT(min_ment=parameters[0],datadirec=datadir)
	cl_inst.load_urls(parameters[1],parameters[2])

	if attribute=='dates':
		assert len(getattr(cl_inst,attribute))==expected and len(getattr(cl_inst,'url_corpus'))==expected
	elif attribute=='url_corpus1':
		mark=True
		for doc in cl_inst.url_corpus:
			for url_ment in doc:
				mark=mark and isinstance(int(url_ment[0]),int)
		assert mark
	elif attribute=='url_corpus2':
		mark=True
		for doc in cl_inst.url_corpus:
			for url_ment in doc:
				mark=mark and isinstance(url_ment[1],str)
		assert mark
	elif attribute=='url_corpus3':
		mark=True
		for doc in cl_inst.url_corpus:
			for url_ment in doc:
				mark=mark and (url_ment[0]>=cl_inst.minimum_ment)
		assert mark
	elif attribute=='url_corpus4':
		cl_inst.load_urls('20130301','20130402')
		assert len(getattr(cl_inst,'dates'))==expected and len(getattr(cl_inst,'url_corpus'))==expected
	return


stinit=[300,'20130901','20130902']
#nonstinit=[300,'20130301','20130401']
test_tokenizer_data=[(stinit,'http://www.fakesite.com/Hail-motherly-Creature','str'),(stinit,'http://www.fakesite.com/Hail-motherly-Creature',['hail','motherli','creatur']),(stinit,'www.fakesite.com/Hail-motherly-Creature',[])]
@pytest.mark.parametrize('parameters,inputs,expected',test_tokenizer_data)
def test_tokenizer(parameters,inputs,expected):
	"""
	Testing tokenizer
	"""
	cl_inst=nlpp.CorpusGDELT(min_ment=parameters[0],datadirec=datadir)
	cl_inst.load_urls(parameters[1],parameters[2])
	if expected=='str':
		mark=True
		for word in cl_inst.url_tokenizer(inputs):
			mark=mark and isinstance(word,str)
		assert mark
	else:
		assert cl_inst.url_tokenizer(inputs)==expected
	return



