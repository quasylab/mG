import importlib.metadata

from spektral.data import Graph
from .compiler import (PsiLocal, PsiGlobal, PsiNonLocal, Phi, Sigma, Constant, Pi, make_uoperator, make_boperator, make_koperator, MGCompiler, CompilerConfig,
                       NodeConfig, EdgeConfig)
from .language import mg_reconstructor, mg_parser
from .data import Dataset, SingleGraphLoader, MultipleGraphLoader
from .explainer import MGExplainer
from .normalizer import mg_normalizer
from .visualizer import print_graph, print_layer

__all__ = ['SingleGraphLoader', 'MultipleGraphLoader', 'PsiLocal', 'PsiGlobal', 'PsiNonLocal', 'Phi', 'Sigma', 'Constant', 'Pi',
           'MGCompiler', 'CompilerConfig', 'NodeConfig', 'EdgeConfig', 'Dataset', 'print_layer', 'print_graph', 'Graph',
           'make_uoperator', 'make_boperator', 'make_koperator', 'MGExplainer', 'mg_normalizer', 'mg_reconstructor', 'mg_parser']

__version__ = importlib.metadata.version('libmg')
