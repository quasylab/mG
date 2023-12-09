"""Defines a normalizer to transform mG expressions in normal form.

This package defines the functions to normalize mG expressions when provided either as strings or expression trees.

The package contains the following objects:

- ``mg_normalizer``
"""
from .normalizer import mg_normalizer

__all__ = ['mg_normalizer']
