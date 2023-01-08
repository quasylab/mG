from libmg.grammar import mg_grammar
import tensorflow as tf
from lark import Lark


class GrammarTests(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.parser = Lark(mg_grammar)

    def test_seq(self):
        expr = 'a;b;c;d;e'
        tree = self.parser.parse(expr)
        self.assertEqual(len(tree.children), 2)
        self.assertEqual(len(tree.children[1].children), 2)
        self.assertEqual(len(tree.children[1].children[1].children), 2)
        self.assertEqual(len(tree.children[1].children[1].children[1].children), 2)

    def test_par(self):
        expr = 'a || b || c || d || e'
        tree = self.parser.parse(expr)
        self.assertEqual(len(tree.children), 5)

    def test_nested_par(self):
        expr = '(a || b) ; c ; d ; ( e || f)'
        tree = self.parser.parse(expr)
        self.assertEqual(len(tree.children), 2)
        self.assertEqual(len(tree.children[0].children), 2)
        self.assertEqual(len(tree.children[1].children), 2)
        self.assertEqual(len(tree.children[1].children[1].children), 2)
        self.assertEqual(len(tree.children[1].children[1].children[1].children), 2)

    def test_nested_seq(self):
        expr = 'a || (b;c) || c || (d;e) || e'
        tree = self.parser.parse(expr)
        self.assertEqual(len(tree.children), 5)

    def test_special_characters(self):
        expr = '<+|*'
        tree = self.parser.parse(expr)
        self.assertEqual(tree.children[0].data, 'function_name')
        self.assertEqual(tree.children[1].data, 'function_name')

        expr = '/;:;|%>&'
        tree = self.parser.parse(expr)
        self.assertEqual(tree.children[0].data, 'fun_app')
        self.assertEqual(tree.children[1].children[0].data, 'fun_app')
        self.assertEqual(tree.children[1].children[1].data, 'rhd')
        self.assertEqual(tree.children[1].children[1].children[0].data, 'function_name')
        self.assertEqual(tree.children[1].children[1].children[1].data, 'function_name')

    def test_var_function_name(self):
        expr = 'add[1]'
        tree = self.parser.parse(expr)
        self.assertEqual(tree.data, 'fun_app')
        self.assertEqual(tree.children[0].data, 'function_name')

        expr = 'add1'
        tree = self.parser.parse(expr)
        self.assertEqual(tree.data, 'fun_app')
        self.assertEqual(tree.children[0].data, 'function_name')

    def test_fun_def(self):
        expr = """
        def test(X:b){
        (f || X);g
        }
        test(h)
        """
        tree = self.parser.parse(expr)
        self.assertEqual(tree.data, 'fun_def')
        self.assertEqual(len(tree.children), 5)
        self.assertEqual(tree.children[0].data, 'function_name')
        self.assertEqual(tree.children[1].data, 'variable_decl')
        self.assertEqual(tree.children[2].data, 'type_decl')
        self.assertEqual(tree.children[4].data, 'fun_call')


if __name__ == '__main__':
    tf.test.main()
