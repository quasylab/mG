"""Defines the parser and reconstructor of the mG language

This package defines the LALR parser, and the (experimental) reconstructor.

The package contains the following objects:

- ``mg_parser``
- ``mg_reconstructor``
"""
from .grammar import mg_parser, mg_reconstructor

__all__ = ['mg_parser', 'mg_reconstructor']
