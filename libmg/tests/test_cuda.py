import sys
import unittest

import tensorflow as tf


class CudaTest(tf.test.TestCase):
    @unittest.skipUnless(sys.platform.startswith('linux'), "Latest TensorFlow versions only support CUDA on GNU/Linux.")
    def test_cuda(self):
        self.assertTrue(tf.test.is_built_with_cuda())
