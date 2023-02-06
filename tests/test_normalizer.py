from libmg.grammar import mg_grammar
import tensorflow as tf
from lark import Lark

from libmg.normalizer import Normalizer


class NormalizerTests(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.parser = Lark(mg_grammar, maybe_placeholders=False)
        self.normalizer = Normalizer(self.parser)

    def test_fixpoint_no_vars(self):
        expr = 'fix X:bool[1] = true in (true || false)'
        tree = self.normalizer.visit(self.parser.parse(expr))
        self.assertEqual('parallel', tree.data,)
        self.assertEqual('atom_op', tree.children[0].data)
        self.assertEqual('atom_op', tree.children[1].data)

        expr = 'fix X:bool[1] = true in (X || false)'
        tree = self.normalizer.visit(self.parser.parse(expr))
        self.assertEqual('fix', tree.data,)
        self.assertEqual('parallel', tree.children[3].data)

    '''
    def test_seq_rhs_fixpoint(self):

        expr = 'fix X:bool[1] = true in (a;X;b)'
        tree = self.normalizer.visit(self.parser.parse(expr))
        self.assertEqual('fix', tree.data)
        self.assertEqual('composition', tree.children[-1].data)

        expr = 'fix X:bool[1] = true in (a;b;X)'
        tree = self.normalizer.visit(self.parser.parse(expr))
        self.assertEqual('fix', tree.data)
        self.assertEqual('atom_op', tree.children[-1].data)

        expr = 'fix X:bool[1] = true in (a;(X || Y))'
        tree = self.normalizer.visit(self.parser.parse(expr))
        self.assertEqual('fix', tree.data)
        self.assertEqual('parallel', tree.children[-1].data)
        self.assertEqual('atom_op', tree.children[-1].children[0].data)
        self.assertEqual('composition', tree.children[-1].children[1].data)

        expr = 'fix X:bool[1] = true in (a;X)'
        tree = self.normalizer.visit(self.parser.parse(expr))
        self.assertEqual('fix', tree.data)
        self.assertEqual('atom_op', tree.children[-1].data)


        expr = 'def f(y:bool[1]){ y } fix X:bool[1] = true in (a;f(X))'
        tree = self.normalizer.visit(self.parser.parse(expr))
        self.assertEqual('fix', tree.children[-1].data)
        self.assertEqual('fun_call', tree.children[-1].children[-1].data)

        expr = 'fix X:bool[1] = true in (a;(let z = b in X;z))'
        tree = self.normalizer.visit(self.parser.parse(expr))
        self.assertEqual('fix', tree.children[-1].data)
        self.assertEqual('fun_call', tree.children[-1].children[-1].data)
        '''


if __name__ == '__main__':
    tf.test.main()
