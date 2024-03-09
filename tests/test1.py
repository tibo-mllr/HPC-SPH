import sys
from testconfig import root_folder
sys.path.append(root_folder)
import pytest
# now the root modules should be available. Make sure you change the root folder to the right path


# @pytest()
def test1():

	assert 1+1==2

@pytest.mark.parametrize('a, b',[(3,2)])
def test2(a,b):
	assert a>b