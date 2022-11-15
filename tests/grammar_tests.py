from sources.grammar import mg_grammar
import tensorflow as tf
from lark import Lark


class GrammarTests(tf.test.TestCase):
    def test_seq(self):
        expr = 'a;b;c;d;e'
        parser = Lark(mg_grammar)
        print(parser.parse(expr).pretty())

    def test_par(self):
        expr = 'a || b || c || d || e'
        parser = Lark(mg_grammar)
        print(parser.parse(expr).pretty())

    def test_nested_par(self):
        expr = '(a || b) ; c ; d ; ( e || f)'
        parser = Lark(mg_grammar)
        print(parser.parse(expr).pretty())

    def test_nested_seq(self):
        expr = 'a || (b;c) || c || (d;e) || e'
        parser = Lark(mg_grammar)
        print(parser.parse(expr).pretty())
