from libmg.grammar import mg_grammar
import tensorflow as tf
from lark import Lark


class GrammarTests(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.parser = Lark(mg_grammar, parser='lalr')

    def test_seq(self):
        expr = 'a;b;c;d;e'
        tree = self.parser.parse(expr)
        self.assertEqual(2, len(tree.children),)
        self.assertEqual(2, len(tree.children[1].children))
        self.assertEqual(2, len(tree.children[1].children[1].children))
        self.assertEqual(2, len(tree.children[1].children[1].children[1].children))

    def test_par(self):
        expr = 'a || b || c || d || e'
        tree = self.parser.parse(expr)
        self.assertEqual(5, len(tree.children))

    def test_nested_par(self):
        expr = '(a || b) ; c ; d ; ( e || f)'
        tree = self.parser.parse(expr)
        self.assertEqual(2, len(tree.children))
        self.assertEqual(2, len(tree.children[0].children))
        self.assertEqual(2, len(tree.children[1].children))
        self.assertEqual(2, len(tree.children[1].children[1].children))
        self.assertEqual(2, len(tree.children[1].children[1].children[1].children))

    def test_nested_seq(self):
        expr = 'a || (b;c) || c || (d;e) || e'
        tree = self.parser.parse(expr)
        self.assertEqual(5, len(tree.children))

    def test_special_characters(self):
        expr = '<+|*'
        tree = self.parser.parse(expr)
        self.assertEqual('label', tree.children[0].data)
        self.assertEqual('label', tree.children[1].data)

        expr = '/;!;|%>&'
        tree = self.parser.parse(expr)
        self.assertEqual('atom_op', tree.children[0].data)
        self.assertEqual('atom_op', tree.children[1].children[0].data)
        self.assertEqual('rhd', tree.children[1].children[1].data)
        self.assertEqual('label', tree.children[1].children[1].children[0].data)
        self.assertEqual('label', tree.children[1].children[1].children[1].data)

    def test_var_function_name(self):
        expr = 'add[1]'
        tree = self.parser.parse(expr)
        self.assertEqual('atom_op', tree.data)
        self.assertEqual('label', tree.children[0].data)

        expr = 'add1'
        tree = self.parser.parse(expr)
        self.assertEqual('atom_op', tree.data)
        self.assertEqual('label', tree.children[0].data)

    def test_uni_fun_def(self):
        expr = """
        def test(X){
        (f || X);g
        } in test(h)
        """
        tree = self.parser.parse(expr)
        self.assertEqual('fun_def', tree.data)
        self.assertEqual(4, len(tree.children))
        self.assertEqual('label_decl', tree.children[0].data)
        self.assertEqual('label_decl', tree.children[1].data)
        self.assertEqual('fun_call', tree.children[3].data)

    def test_mul_fun_def(self):
        expr = """
        def test(X, Y){
        (Y || X);g
        } in test(h, f)
        """
        tree = self.parser.parse(expr)
        self.assertEqual('fun_def', tree.data)
        self.assertEqual(5, len(tree.children))
        self.assertEqual('label_decl', tree.children[0].data)
        self.assertEqual('label_decl', tree.children[1].data,)
        self.assertEqual('label_decl', tree.children[2].data)
        self.assertEqual('fun_call', tree.children[4].data)

    def test_var_def(self):
        expr = """
        let test = b;not in test
        """
        tree = self.parser.parse(expr)
        self.assertEqual('local_var_expr', tree.data)
        self.assertEqual(3, len(tree.children))
        self.assertEqual('label_decl', tree.children[0].data)
        self.assertEqual('atom_op', tree.children[2].data)
        expr = """
        let t1 = b;not, t2 = f(x, y) in t1 || t2
        """
        tree = self.parser.parse(expr)
        self.assertEqual('local_var_expr', tree.data)
        self.assertEqual(5, len(tree.children))
        self.assertEqual('label_decl', tree.children[0].data)
        self.assertEqual('label_decl', tree.children[2].data)
        self.assertEqual('parallel', tree.children[4].data)

    def test_local_var_expr(self):
        expr = """
        let x = a, y = b, z = true in
        (x || y);z
        """
        tree = self.parser.parse(expr)
        self.assertEqual('local_var_expr', tree.data)
        self.assertEqual('label_decl', tree.children[0].data)
        self.assertEqual('atom_op', tree.children[1].data)
        self.assertEqual('label_decl', tree.children[2].data)
        self.assertEqual('atom_op', tree.children[3].data)
        self.assertEqual('label_decl', tree.children[4].data)
        self.assertEqual('atom_op', tree.children[5].data)
        self.assertEqual('composition', tree.children[6].data)

    def test_ite(self):
        expr = """
        if a then
            b
        else
            false
        """
        tree = self.parser.parse(expr)
        self.assertEqual('ite', tree.data)
        self.assertEqual('atom_op', tree.children[0].data)
        self.assertEqual('atom_op', tree.children[1].data)
        self.assertEqual('atom_op', tree.children[2].data)

    def test_fix(self):
        expr = 'fix x = true in (a || (x;|>or));and'
        tree = self.parser.parse(expr)
        self.assertEqual('fix', tree.data)
        self.assertEqual('label_decl', tree.children[0].data)
        self.assertEqual('atom_op', tree.children[1].data)
        self.assertEqual('composition', tree.children[2].data)

    def test_comment(self):
        expr = '# this is a comment\n a'
        tree = self.parser.parse(expr)
        self.assertEqual('atom_op', tree.data)


if __name__ == '__main__':
    tf.test.main()
