import pytest
import pathlib


def run_tests():
    pytest.main([pathlib.Path(__file__).parent.parent.resolve()])
