"""
This is a script for testing functions and methods in the nlp preprocessing module
"""
import nlp_preprocessing as nlpp
import pytest


test_class_init_data=[([-2],'minimum_ment',1),([2],'minimum_ment',2),([2],'currentdir','data/GDELT_1.0/')]
@pytest.mark.parametrize('parameters,attribute,expected',test_class_init_data)
def test_class_init(parameters,attribute,expected):
	"""
	Very beginning of testing of the initializer, bear with me while I expand this module
	"""
	if len(parameters)==1:
		cl_inst=nlpp.CorpusGDELT(min_ment=parameters[0])
	else:
		cl_inst=nlpp.CorpusGDELT(min_ment=parameters[0],datadirec=parameters[1])
	assert getattr(cl_inst,attribute)==expected