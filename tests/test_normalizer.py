
import tensorflow as tf

from libmg.grammar import mg_parser
from libmg.normalizer import Normalizer


class NormalizerTests(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.parser = mg_parser
        self.normalizer = Normalizer()

    def test_fixpoint_no_vars(self):
        expr = 'fix X = true in (true || false)'
        tree = self.normalizer.visit(self.parser.parse(expr))
        self.assertEqual('parallel', tree.data,)
        self.assertEqual('atom_op', tree.children[0].data)
        self.assertEqual('atom_op', tree.children[1].data)

        expr = 'fix X = true in (X || false)'
        tree = self.normalizer.visit(self.parser.parse(expr))
        self.assertEqual('fix', tree.data,)
        self.assertEqual('parallel', tree.children[2].data)

    def test_composition_fixvars(self):
        expr = ['fix X = true in X;X', 'fix X = true in X;X;not', 'fix X = true in X;X;X', 'fix X = true in X;not;X',
                'fix X = true in (X;(not || true || X);or)', 'fix X = true in (X;if X then true else false)',
                'fix X = true in (X;if true then X else false)', 'fix X = true in (a ; ((X || not);and))']
        eq_expr = ['fix X = true in X', 'fix X = true in X;not', 'fix X = true in X', 'fix X = true in X',
                   'fix X = true in (((X;not) || (X;true) || X);or)',
                   'fix X = true in (if X then (X;true) else (X;false))',
                   'fix X = true in (if (X;true) then X else (X;false))',
                   'fix X = true in ((X || (a;not));and)']
        for e1, e2 in zip(expr, eq_expr):
            tree1 = self.normalizer.visit(self.parser.parse(e1))
            tree2 = self.parser.parse(e2)
            self.assertEqual(tree1, tree2)


if __name__ == '__main__':
    tf.test.main()
