
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


if __name__ == '__main__':
    tf.test.main()
