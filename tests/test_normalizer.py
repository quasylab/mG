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
        expr = 'fix X = true in (true || false)'
        tree = self.normalizer.visit(self.parser.parse(expr))
        self.assertEqual('parallel', tree.data,)
        self.assertEqual('atom_op', tree.children[0].data)
        self.assertEqual('atom_op', tree.children[1].data)

        expr = 'fix X = true in (X || false)'
        tree = self.normalizer.visit(self.parser.parse(expr))
        self.assertEqual('fix', tree.data,)
        self.assertEqual('parallel', tree.children[2].data)


if __name__ == '__main__':
    tf.test.main()
