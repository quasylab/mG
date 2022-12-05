from libmg import PsiLocal, FunctionDict
import tensorflow as tf


class DictionaryTest(tf.test.TestCase):
    def test_simple_function(self):
        my_dict = FunctionDict({'add': PsiLocal(tf.math.add), 'sub': PsiLocal(tf.math.subtract)})
        self.assertIsInstance(my_dict['add'], PsiLocal)
        self.assertIsInstance(my_dict['sub'], PsiLocal)

    def test_function_generator(self):
        my_dict = FunctionDict({})
        my_dict['add'] = lambda x: PsiLocal(lambda y: tf.math.add(y, int(x[-1])))
        self.assertIsInstance(my_dict['add[1]'], PsiLocal)
        self.assertEqual(my_dict['add[1]'](tf.constant(1)), tf.constant(2))


if __name__ == '__main__':
    tf.test.main()
