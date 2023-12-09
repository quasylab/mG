"""Defines the mG compiler and the basic mG functions.

This package defines a compiler for mG programs and the data structures to instantiate it. It also provides the various classes that allow for the definition
of mG functions.

The package contains the following classes:

- ``NodeConfig``
- ``EdgeConfig``
- ``CompilerConfig``
- ``MGCompiler``
- ``PsiNonLocal``
- ``PsiLocal``
- ``PsiGlobal``
- ``Phi``
- ``Sigma``
- ``Constant``
- ``Pi``

The module contains the following functions:

- ``make_uoperator(op, name)``
- ``make_boperator(op, name)``
- ``make_koperator(op, name)``
"""
from .functions import PsiLocal, PsiGlobal, PsiNonLocal, Phi, Sigma, Constant, Pi, make_uoperator, make_boperator, make_koperator
from .compiler import NodeConfig, EdgeConfig, CompilerConfig, MGCompiler

__all__ = ['PsiLocal', 'PsiGlobal', 'PsiNonLocal', 'Phi', 'Sigma', 'Constant', 'Pi', 'make_uoperator', 'make_boperator', 'make_koperator',
           'CompilerConfig', 'NodeConfig', 'EdgeConfig', 'MGCompiler']
