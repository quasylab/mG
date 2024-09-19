"""Defines a visualizer for graphs.

This package defines the functions to view graphs and mG model outputs on a web browser in an interactive way.

The package contains the following functions:

- ``print_graph(graph, node_names_func='id', hierarchical=False, show_labels=False, open_browser=True)``
- ``print_layer(model, inputs, labels=None, layer_name=None, layer_idx=None, open_browser=True)``
"""
from .visualizer import print_graph, print_layer, print_labels

__all__ = ['print_graph', 'print_layer', 'print_labels']
