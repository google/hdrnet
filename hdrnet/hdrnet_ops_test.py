# Lint as: python3
"""Tests for custom tensorflow operators in HDRnet (CUDA only)."""

import collections

import hdrnet_ops as ops
import numpy as np
from parameterized import parameterized
import tensorflow.compat.v1 as tf


def _assert_tf_shape_equals(test_case, expected_shape, tf_tensor):
  tf_shape = tf_tensor.shape.as_list()
  tf.logging.info('expected_shape: %s, tf_shape: %s', expected_shape, tf_shape)
  test_case.assertEqual(expected_shape, tf_shape)


def _assert_np_shape_equals(test_case, expected_shape, np_array):
  np_shape = list(np_array.shape)
  tf.logging.info('expected_shape: %s, np_shape: %s', expected_shape, np_shape)
  test_case.assertEqual(expected_shape, np_shape)


def _assert_shape_equals(test_case, expected_shape, np_array, tf_tensor):
  """Asserts equality on shapes given numpy and tf inputs.

  Asserts (using test_case.assertEqual) that both np_array and tf_tensor has
  shape expected_shape.

  Args:
    test_case: (tf.test.TestCase) The TestCase currently being run.
    expected_shape: (list of ints) The expected shape.
    np_array: (np.ndarray) The input numpy array.
    tf_tensor: (tf.Tensor) The input Tensorflow Tensor.
  """

  _assert_tf_shape_equals(test_case, expected_shape, tf_tensor)
  _assert_np_shape_equals(test_case, expected_shape, np_array)


def _get_device_string(use_gpu):
  if use_gpu:
    return '/gpu:0'
  else:
    return '/cpu:0'


_ForwardTestExtents = collections.namedtuple(
    '_ForwardTestExtents',
    'batch_size, h, w, input_channels, gh, gw, gd, gc, output_channels')


class BilateralSliceTest(tf.test.TestCase):

  def run_bilateral_slice(self, grid_data, guide_data, use_gpu):
    dev = _get_device_string(use_gpu)

    graph = tf.Graph()
    with graph.as_default():
      with tf.device(dev):
        grid_tensor = tf.convert_to_tensor(
            grid_data, name='grid', dtype=tf.float32)
        guide_tensor = tf.convert_to_tensor(
            guide_data, name='guide', dtype=tf.float32)
        output_tensor = ops.bilateral_slice(grid_tensor, guide_tensor)
      with self.test_session(
          graph=graph, use_gpu=use_gpu, force_gpu=use_gpu) as sess:
        output_data = sess.run(output_tensor)
    return output_data, output_tensor

  def run_bilateral_slice_grad(self, grid_data, guide_data, use_gpu):
    dev = _get_device_string(use_gpu)

    graph = tf.Graph()
    with graph.as_default():
      with tf.device(dev):
        grid_tensor = tf.convert_to_tensor(
            grid_data, name='grid', dtype=tf.float32)
        guide_tensor = tf.convert_to_tensor(
            guide_data, name='guide', dtype=tf.float32)
        output_tensor = ops.bilateral_slice(grid_tensor, guide_tensor)
        grid_grad_tensor, guide_grad_tensor = tf.gradients(
            output_tensor, [grid_tensor, guide_tensor])
        with self.test_session(
            graph=graph, use_gpu=use_gpu, force_gpu=use_gpu) as sess:
          grid_grad_data, guide_grad_data = sess.run(
              [grid_grad_tensor, guide_grad_tensor])

    return (grid_grad_tensor, guide_grad_tensor, grid_grad_data,
            guide_grad_data)

  def create_forward_test(self,
                          batch_size=3,
                          h=30,
                          w=25,
                          input_channels=3,
                          gh=16,
                          gw=12,
                          gd=8,
                          output_channels=3,
                          randomize_values=True):
    np.random.seed(1234)
    gc = output_channels * (1 + input_channels)
    grid_shape = (batch_size, gh, gw, gd, gc)
    guide_shape = (batch_size, h, w)
    if randomize_values:
      grid_data = np.random.rand(*grid_shape).astype(np.float32)
      guide_data = np.random.rand(*guide_shape).astype(np.float32)
    else:
      grid_data = np.zeros(grid_shape)
      guide_data = np.zeros(guide_shape)
    sz = _ForwardTestExtents(batch_size, h, w, input_channels, gh, gw, gd, gc,
                             output_channels)
    return sz, grid_data, guide_data

  @parameterized.expand([('CPU', False), ('GPU', True)])
  def test_shape(self, use_gpu):
    """bilateral_slice(grid, guide) should have the right shape."""
    sz, grid_data, guide_data = self.create_forward_test()
    output_data, output_tensor = self.run_bilateral_slice(
        grid_data, guide_data, use_gpu)

    _assert_shape_equals(self, [sz.batch_size, sz.h, sz.w, sz.gc], output_data,
                         output_tensor)

  @parameterized.expand([('CPU', False), ('GPU', True)])
  def test_grad_shape(self, use_gpu):
    """gradient(bilateral_slice(grid, guide)) should have the right shape."""
    sz, grid_data, guide_data = self.create_forward_test()
    grid_grad_tensor, guide_grad_tensor, grid_grad_data, guide_grad_data = (
        self.run_bilateral_slice_grad(grid_data, guide_data, use_gpu))

    _assert_shape_equals(self, [sz.batch_size, sz.gh, sz.gw, sz.gd, sz.gc],
                         grid_grad_data, grid_grad_tensor)
    _assert_shape_equals(self, [sz.batch_size, sz.h, sz.w], guide_grad_data,
                         guide_grad_tensor)

  # TODO(jiawen): Read back both CPU and GPU gradients and compare them to each
  # other as well as the gradient checker.
  def run_grad_test(self, batch_size, h, w, input_channels, gh, gw, gd,
                    output_channels, grad_tensor_name, use_gpu):
    dev = _get_device_string(use_gpu)

    gc = (1 + input_channels) * output_channels
    grid_shape = [batch_size, gh, gw, gd, gc]
    guide_shape = [batch_size, h, w]
    output_shape = [batch_size, h, w, gc]

    grid_data = np.random.rand(*grid_shape).astype(np.float32)
    guide_data = np.random.rand(*guide_shape).astype(np.float32)

    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
      with tf.device(dev):
        grid_tensor = tf.convert_to_tensor(
            grid_data, name='grid', dtype=tf.float32)
        guide_tensor = tf.convert_to_tensor(
            guide_data, name='guide', dtype=tf.float32)

        output_tensor = ops.bilateral_slice(grid_tensor, guide_tensor)

        if grad_tensor_name == 'grid':
          grad_tensor = grid_tensor
          grad_shape = grid_shape
        elif grad_tensor_name == 'guide':
          grad_tensor = guide_tensor
          grad_shape = guide_shape

        # It is important to use self.test_session, which will disable the
        # graph optimization, otherwise it won't use GPU ops. See details here:
        # https://github.com/tensorflow/tensorflow/issues/2054
        with self.test_session(graph=graph, use_gpu=use_gpu, force_gpu=use_gpu):
          err = tf.test.compute_gradient_error(
              grad_tensor, grad_shape, output_tensor, output_shape, delta=1e-4)
        # Note that the gradient cannot be accurate, as trilinear interpolation
        # is not a smooth function. When the interpolated point is on the grid,
        # the gradient does not exist. Therefore, the analytical gradient (by
        # gradient op, implemented in bilateral_slice.cu.cc) and numerical
        # grident (by tf.test.compute_gradient_error) will never match.
        self.assertLess(err, 3e-3)

  @parameterized.expand([('CPU', False), ('GPU', True)])
  def test_grid_gradient(self, use_gpu):
    """True grid derivative should closely match numerical derivative."""
    self.run_grad_test(
        batch_size=3,
        h=8,
        w=5,
        input_channels=3,
        gh=6,
        gw=3,
        gd=7,
        output_channels=4,
        use_gpu=use_gpu,
        grad_tensor_name='grid')

  @parameterized.expand([('CPU', False), ('GPU', True)])
  def test_guide_gradient(self, use_gpu):
    """True guide derivative should closely match numerical derivative."""
    self.run_grad_test(
        batch_size=1,
        h=6,
        w=18,
        input_channels=1,
        gh=3,
        gw=9,
        gd=7,
        output_channels=1,
        use_gpu=use_gpu,
        grad_tensor_name='guide')


class BilateralSliceApplyTest(tf.test.TestCase):

  def run_bilateral_slice_apply(self,
                                grid_data,
                                guide_data,
                                input_data,
                                use_gpu=False):
    dev = _get_device_string(use_gpu)

    graph = tf.Graph()
    with graph.as_default():
      with tf.device(dev):
        grid_tensor = tf.convert_to_tensor(
            grid_data, name='grid', dtype=tf.float32)
        guide_tensor = tf.convert_to_tensor(
            guide_data, name='guide', dtype=tf.float32)
        input_tensor = tf.convert_to_tensor(
            input_data, name='input', dtype=tf.float32)
        output_tensor = ops.bilateral_slice_apply(
            grid_tensor, guide_tensor, input_tensor, has_offset=True)
      with self.test_session(
          graph=graph, use_gpu=use_gpu, force_gpu=use_gpu) as sess:
        output_data = sess.run(output_tensor)

    return output_data, output_tensor

  def run_bilateral_slice_apply_grad(self,
                                     grid_data,
                                     guide_data,
                                     input_data,
                                     use_gpu=False):
    dev = _get_device_string(use_gpu)

    graph = tf.Graph()
    with graph.as_default():
      with tf.device(dev):
        grid_tensor = tf.convert_to_tensor(
            grid_data, name='grid', dtype=tf.float32)
        guide_tensor = tf.convert_to_tensor(
            guide_data, name='guide', dtype=tf.float32)
        input_tensor = tf.convert_to_tensor(
            input_data, name='input', dtype=tf.float32)
        output_tensor = ops.bilateral_slice_apply(
            grid_tensor, guide_tensor, input_tensor, has_offset=True)
        grid_grad_tensor, guide_grad_tensor, _ = tf.gradients(
            output_tensor, [grid_tensor, guide_tensor, input_tensor])
        with self.test_session(
            graph=graph, use_gpu=use_gpu, force_gpu=use_gpu) as sess:
          grid_grad_data, guide_grad_data = sess.run(
              [grid_grad_tensor, guide_grad_tensor])

    return (grid_grad_tensor, guide_grad_tensor, grid_grad_data,
            guide_grad_data)

  def create_forward_test(self,
                          batch_size=3,
                          h=30,
                          w=25,
                          input_channels=3,
                          gh=16,
                          gw=12,
                          gd=8,
                          output_channels=3,
                          randomize_values=True):
    np.random.seed(1234)
    gc = output_channels * (1 + input_channels)
    grid_shape = (batch_size, gh, gw, gd, gc)
    guide_shape = (batch_size, h, w)
    input_shape = (batch_size, h, w, input_channels)
    if randomize_values:
      grid_data = np.random.rand(*grid_shape).astype(np.float32)
      guide_data = np.random.rand(*guide_shape).astype(np.float32)
      input_data = np.random.rand(*input_shape).astype(np.float32)
    else:
      grid_data = np.zeros(grid_shape)
      guide_data = np.zeros(guide_shape)
      input_data = np.zeros(input_shape)
    sz = _ForwardTestExtents(batch_size, h, w, input_channels, gh, gw, gd, gc,
                             output_channels)
    return sz, grid_data, guide_data, input_data

  @parameterized.expand([('CPU', False), ('GPU', True)])
  def test_shape(self, use_gpu):
    """bilateral_slice_apply(grid, guide) should have the right shape."""
    sz, grid_data, guide_data, input_data = self.create_forward_test()

    output_data, output_tensor = self.run_bilateral_slice_apply(
        grid_data, guide_data, input_data, use_gpu)
    _assert_shape_equals(self, [sz.batch_size, sz.h, sz.w, sz.output_channels],
                         output_data, output_tensor)

  @parameterized.expand([('CPU', False), ('GPU', True)])
  def test_grad_shape(self, use_gpu):
    """grad(bilateral_slice_apply(grid, guide)) should have the right shape."""
    sz, grid_data, guide_data, input_data = self.create_forward_test()

    grid_grad_tensor, guide_grad_tensor, grid_grad_data, guide_grad_data = (
        self.run_bilateral_slice_apply_grad(grid_data, guide_data, input_data,
                                            use_gpu))
    _assert_shape_equals(self, [sz.batch_size, sz.gh, sz.gw, sz.gd, sz.gc],
                         grid_grad_data, grid_grad_tensor)
    _assert_shape_equals(self, [sz.batch_size, sz.h, sz.w], guide_grad_data,
                         guide_grad_tensor)

  # TODO(jiawen): Read back both CPU and GPU gradients and compare them to each
  # other as well as the gradient checker.
  def run_grad_test(self, batch_size, h, w, input_channels, gh, gw, gd,
                    output_channels, use_gpu, grad_tensor_name):
    grid_shape = (batch_size, gh, gw, gd,
                  (1 + input_channels) * output_channels)
    guide_shape = (batch_size, h, w)
    input_shape = (batch_size, h, w, input_channels)
    output_shape = (batch_size, h, w, output_channels)

    grid_data = np.random.rand(*grid_shape).astype(np.float32)
    guide_data = np.random.rand(*guide_shape).astype(np.float32)
    input_data = np.random.rand(*input_shape).astype(np.float32)

    dev = _get_device_string(use_gpu)

    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
      with tf.device(dev):
        grid_tensor = tf.convert_to_tensor(
            grid_data, name='grid', dtype=tf.float32)
        guide_tensor = tf.convert_to_tensor(
            guide_data, name='guide', dtype=tf.float32)
        input_tensor = tf.convert_to_tensor(
            input_data, name='input', dtype=tf.float32)

        output_tensor = ops.bilateral_slice_apply(
            grid_tensor, guide_tensor, input_tensor, has_offset=True)

        if grad_tensor_name == 'grid':
          grad_tensor = grid_tensor
          grad_shape = grid_shape
        elif grad_tensor_name == 'guide':
          grad_tensor = guide_tensor
          grad_shape = guide_shape
        elif grad_tensor_name == 'input':
          grad_tensor = input_tensor
          grad_shape = input_shape

        # It is important to use self.test_session, which will disable the
        # graph optimization, otherwise it won't use GPU ops. See details here:
        # https://github.com/tensorflow/tensorflow/issues/2054
        with self.test_session(graph=graph, use_gpu=use_gpu, force_gpu=use_gpu):
          err = tf.test.compute_gradient_error(grad_tensor, grad_shape,
                                               output_tensor, output_shape)
        self.assertLess(err, 1e-2)

  @parameterized.expand([('CPU', False), ('GPU', True)])
  def test_grid_gradient(self, use_gpu):
    """True grid derivative should closely match numerical derivative."""
    self.run_grad_test(
        batch_size=3,
        h=8,
        w=5,
        input_channels=3,
        gh=6,
        gw=3,
        gd=7,
        output_channels=4,
        use_gpu=use_gpu,
        grad_tensor_name='grid')

  @parameterized.expand([('CPU', False), ('GPU', True)])
  def test_guide_gradient(self, use_gpu):
    """True guide derivative should closely match numerical derivative."""
    self.run_grad_test(
        batch_size=1,
        h=6,
        w=18,
        input_channels=1,
        gh=3,
        gw=9,
        gd=7,
        output_channels=1,
        use_gpu=use_gpu,
        grad_tensor_name='guide')

  @parameterized.expand([('CPU', False), ('GPU', True)])
  def test_input_gradient(self, use_gpu):
    """True input derivative should closely match numerical derivative."""
    self.run_grad_test(
        batch_size=3,
        h=8,
        w=5,
        input_channels=3,
        gh=6,
        gw=3,
        gd=7,
        output_channels=4,
        use_gpu=use_gpu,
        grad_tensor_name='input')


if __name__ == '__main__':
  tf.test.main()
