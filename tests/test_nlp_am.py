import nlp_preprocessing as nlpp
import pytest
"""
This is a script for testing functions and methods in the nlp preprocessing module
"""

test_class_init_data=[([-2],'minimum_ment',1),]
@pytest.mark.parametrize('parameters,attribute,expected',test_class_init_data)
def test_class_init(parameters,attribute,expected):
	if len(parameters)==1:
		cl_inst=nlpp.CorpusGDELT(min_ment=parameters[0])
	else:
		cl_inst=nlpp.CorpusGDELT(min_ment=parameters[0],datadiec=parameters[1])
	assert getattr(cl_inst,attribute)==expected

