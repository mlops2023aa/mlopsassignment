"""Module docstring explaining the purpose of the module."""

import math
import sys
sys.path.append('mlopsassignment/models/')
#from mlopsassignment/models/randomforest import add_numbers


def test_add_numbers():
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0