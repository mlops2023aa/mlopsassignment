"""Module docstring explaining the purpose of the module."""


from models.randomforest import add_numbers
# import sys
# sys.path.append('mlopsassignment/models/')


def test_add_numbers():
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
