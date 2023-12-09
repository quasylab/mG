"""Defines an explainer for mG models.

This package defines the means to generate the sub-graph of all nodes that influenced the final label of some query node.

The package contains the following classes:

- ``MGExplainer``
"""
from .explainer import MGExplainer

__all__ = ['MGExplainer']
