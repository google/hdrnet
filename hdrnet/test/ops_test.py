# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test for custom tensorflow operators."""

import os
import unittest
import tempfile

import numpy as np
import skimage.color as skcolor
import tensorflow as tf

import hdrnet.hdrnet_ops as ops


class BilateralSliceTest(tf.test.TestCase):
  def run_bilateral_slice(self, dev, grid_data, guide_data):
    with tf.device(dev):

      grid_tensor = tf.convert_to_tensor(
          grid_data, name='grid', dtype=tf.float32)
      guide_tensor = tf.convert_to_tensor(
          guide_data, name='guide', dtype=tf.float32)

      output_tensor = ops.bilateral_slice(grid_tensor, guide_tensor)

    with self.test_session() as sess:
      output_data = sess.run(output_tensor)

    return output_data

  def test_shape_is_correct(self):
    batch_size = 3
    grid_shape = [batch_size, 10, 6, 8, 12]
    guide_shape = [batch_size, 101, 60]
    grid_data = np.random.rand(*grid_shape).astype(np.float32)
    guide_data = np.random.rand(*guide_shape).astype(np.float32)

    for dev in ['/cpu:0', '/gpu:0']:
      output_data = self.run_bilateral_slice(dev, grid_data, guide_data)
      output_shape = list(output_data.shape)

      self.assertEqual(len(output_shape), 4)
      self.assertEqual(output_shape[0], guide_shape[0])
      self.assertEqual(output_shape[1], guide_shape[1])
      self.assertEqual(output_shape[2], guide_shape[2])
      self.assertEqual(output_shape[3], grid_shape[4])

  def test_interpolate(self):
    for dev in ['/gpu:0']:
      batch_size = 3
      h = 3
      w = 4
      d = 3
      grid_shape = [batch_size, h, w, d, 1]
      grid_data = np.zeros(grid_shape).astype(np.float32)
      grid_data[:, :, :, 1 :] = 1.0
      grid_data[:, :, :, 2 :] = 2.0

      guide_shape = [batch_size, 5, 9]
      target_shape = [batch_size, 5, 9, 1]

      for val in range(d):
        target_data = val*np.ones(target_shape)
        target_data = target_data.astype(np.float32)

        guide_data = ((val+0.5)/(1.0*d))*np.ones(guide_shape).astype(np.float32)
        output_data = self.run_bilateral_slice(dev, grid_data, guide_data)
        diff = np.amax(np.abs(target_data-output_data))


        self.assertEqual(target_shape, list(output_data.shape))

        self.assertLess(diff, 5e-4)

  def test_grid_gradient(self):
    for dev in ['/gpu:0']:
      batch_size = 3
      h = 8
      w = 5
      gh = 6
      gw = 3
      d = 7
      nchans = 4
      grid_shape = [batch_size, gh, gw, d, nchans]
      guide_shape = [batch_size, h, w]
      output_shape = [batch_size, h, w, nchans]
      grid_data = np.random.rand(*grid_shape).astype(np.float32)
      guide_data = np.random.rand(*guide_shape).astype(np.float32)

      with tf.device(dev):
        grid_tensor = tf.convert_to_tensor(grid_data,
                                           name='data',
                                           dtype=tf.float32)
        guide_tensor = tf.convert_to_tensor(guide_data,
                                            name='data',
                                            dtype=tf.float32)

        output_tensor = ops.bilateral_slice(grid_tensor, guide_tensor)

      with self.test_session():
        err = tf.test.compute_gradient_error(
            grid_tensor,
            grid_shape,
            output_tensor,
            output_shape)

        self.assertLess(err, 1e-4)

  def test_guide_gradient(self):
    for dev in ['/gpu:0']:
      batch_size = 2
      h = 7
      w = 8
      d = 5
      gh = 3
      gw = 4
      nchans = 2
      grid_shape = [batch_size, gh, gw, d, nchans]
      guide_shape = [batch_size, h, w]
      output_shape = [batch_size, h, w, nchans]
      grid_data = np.random.randn(*grid_shape).astype(np.float32)
      guide_data = np.random.rand(*guide_shape).astype(np.float32)

      with tf.device(dev):
        grid_tensor = tf.convert_to_tensor(grid_data,
                                           name='data',
                                           dtype=tf.float32)
        guide_tensor = tf.convert_to_tensor(guide_data,
                                            name='data',
                                            dtype=tf.float32)

        output_tensor = ops.bilateral_slice(grid_tensor, guide_tensor)

      with self.test_session():
        th, num = tf.test.compute_gradient(
            guide_tensor,
            guide_shape,
            output_tensor,
            output_shape, delta=1e-4)

        print th
        print num

        thresh = 5e-3
        diff = np.abs(th-num)
        x, y = np.where(diff>thresh)
        for i in range(len(x)):
          in_x = x[i] % w
          in_y = x[i] / w
          out_c = y[i] % nchans
          out_x = (y[i]/nchans) % w
          out_y = (y[i]/nchans) / w
          print "output ({},{},{}) - input ({},{})\n  guide: {:f}\n  theoretical: {:f}\n  numerical: {:f}".format(
              out_y, out_x, out_c, in_y, in_x, np.ravel(guide_data)[x[i]], th[x[i], y[i]], num[x[i],y[i]])

        print len(x), 'of', len(np.ravel(diff)), 'errors'

        print 'gradient shape', th.shape
        print 'guide shape', guide_data.shape
        print 'grid shape', grid_data.shape
        print 'output shape', output_shape

        self.assertLess(np.amax(diff), thresh)


  def l2_optimizer(self, target, output, lr=1e-2):
    loss = tf.reduce_sum(tf.square(target-output))
    global_step = tf.Variable(
        0, name='global_step', trainable=False,
        collections=['global_step', tf.GraphKeys.GLOBAL_VARIABLES])
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(
        loss, global_step=global_step)

    return optimizer, loss

  def test_grid_optimize(self):
    for dev in ['/gpu:0']:
      bs= 1
      h = 1
      w = 32
      nchans = 1
      gh = 1
      gw = 16
      gd = 8

      guide_data = np.linspace(0, 1, w).astype(np.float32)
      guide_data = guide_data[np.newaxis, np.newaxis, :]
      guide_data = np.tile(guide_data, [bs, h, 1])

      grid_data = np.random.rand(bs, gh, gw, gd, nchans).astype(np.float32)

      target_data = np.sin(np.linspace(0, 2*np.pi, w)).astype(np.float32)
      target_data = target_data[np.newaxis, np.newaxis, :, np.newaxis]
      target_data = np.tile(target_data, [bs, h, 1, 1])

      grid = tf.Variable(grid_data)
      guide = tf.convert_to_tensor(guide_data)
      target = tf.convert_to_tensor(target_data)

      output = ops.bilateral_slice(grid, guide)

      checkpoint_dir = tempfile.mkdtemp()

      opt, loss = self.l2_optimizer(target, output)

      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        for step in range(10000):
          _, l_ = sess.run([opt, loss])
          if step % 100 == 0:
            print "Step {}, loss = {:.5f}".format(step, l_)

        out_, target_ = sess.run([output, target])
        out_ = np.squeeze(out_)
        target_ = np.squeeze(target_)

        assert np.sum(np.square(out_-target_)) < 0.0085

  def test_guide_optimize(self):
    for dev in ['/gpu:0']:
      bs= 1
      h = 1
      w = 32
      nchans = 1
      gh = 1
      gw = 8
      gd = 2

      guide_data = np.linspace(0.5/gd, 1-0.5/gd, w).astype(np.float32)
      guide_data = guide_data[np.newaxis, np.newaxis, :]
      guide_data = np.tile(guide_data, [bs, h, 1])

      grid_data = np.linspace(-1, 1, gd).astype(np.float32)
      grid_data = grid_data[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis ]
      grid_data = np.tile(grid_data, [bs, gh, gw, 1, nchans])

      target_data = np.sin(np.linspace(0, 2*np.pi, w)).astype(np.float32)
      target_data = target_data[np.newaxis, np.newaxis, :, np.newaxis]
      target_data = np.tile(target_data, [bs, h, 1, 1])

      grid = tf.convert_to_tensor(grid_data)
      guide = tf.Variable(guide_data)
      guide = tf.sigmoid(guide)
      target = tf.convert_to_tensor(target_data)

      output = ops.bilateral_slice(grid, guide)

      checkpoint_dir = tempfile.mkdtemp()

      opt, loss = self.l2_optimizer(target, output, lr=1e-3)

      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        for step in range(6000):
          _, l_ = sess.run([opt, loss])
          if step % 100 == 0:
            print "Step {}, loss = {:.5f}".format(step, l_)

        out_, target_, guide_, grid_ = sess.run([output, target, guide, grid])
        out_ = np.squeeze(out_)
        target_ = np.squeeze(target_)
        guide_ = np.squeeze(guide_)
        grid_ = np.squeeze(grid_)

        assert np.sum(np.square(out_-target_)) < 1e-4

  def test_optimize_both(self):
    for dev in ['/gpu:0']:
      bs= 1
      h = 1
      w = 32
      nchans = 1
      gh = 1
      gw = 8
      gd = 2

      guide_data = np.random.rand(bs, h, w).astype(np.float32)*2.0-1.0

      grid_data = np.random.rand(bs, gh, gw, gd, nchans).astype(np.float32)

      target_data = np.sin(np.linspace(0, 2*np.pi, w)).astype(np.float32)
      target_data = target_data[np.newaxis, np.newaxis, :, np.newaxis]
      target_data = np.tile(target_data, [bs, h, 1, 1])

      grid = tf.Variable(grid_data)
      guide = tf.Variable(guide_data)
      guide = tf.sigmoid(guide)
      target = tf.convert_to_tensor(target_data)

      output = ops.bilateral_slice(grid, guide)

      checkpoint_dir = tempfile.mkdtemp()

      opt, loss = self.l2_optimizer(target, output, lr=1e-1)

      with self.test_session() as sess:
        tf.global_variables_initializer().run()
        for step in range(10000):
          _, l_ = sess.run([opt, loss])
          if step % 100 == 0:
            print "Step {}, loss = {:.5f}".format(step, l_)

        out_, target_, guide_, grid_ = sess.run([output, target, guide, grid])
        out_ = np.squeeze(out_)
        target_ = np.squeeze(target_)
        guide_ = np.squeeze(guide_)
        grid_ = np.squeeze(grid_)

        assert np.sum(np.square(out_-target_)) < 1e-4




class BilateralSliceApplyTest(tf.test.TestCase):
  def run_bilateral_slice_apply(self, dev, grid_data, guide_data, input_data, has_offset=False):
    with tf.device(dev):

      grid_tensor = tf.convert_to_tensor(
          grid_data, name='grid', dtype=tf.float32)
      guide_tensor = tf.convert_to_tensor(
          guide_data, name='guide', dtype=tf.float32)
      input_tensor = tf.convert_to_tensor(
          input_data, name='input', dtype=tf.float32)

      output_tensor = ops.bilateral_slice_apply(grid_tensor, guide_tensor, input_tensor, has_offset=has_offset)

    with self.test_session() as sess:
      output_data = sess.run(output_tensor)

    return output_data

  def test_shape_is_correct(self):
    batch_size = 3
    grid_shape = [batch_size, 10, 6, 8, 12]
    guide_shape = [batch_size, 101, 60]
    input_shape = [batch_size, 101, 60, 3]
    grid_data = np.random.rand(*grid_shape).astype(np.float32)
    guide_data = np.random.rand(*guide_shape).astype(np.float32)
    input_data = np.random.rand(*input_shape).astype(np.float32)

    for dev in ['/gpu:0']:
      output_data = self.run_bilateral_slice_apply(dev, grid_data, guide_data, input_data, has_offset=True)
      output_data_no_offset = self.run_bilateral_slice_apply(dev, grid_data, guide_data, input_data, has_offset=False)
      output_shape = list(output_data.shape)
      output_shape_no_offset = list(output_data_no_offset.shape)

      self.assertEqual(len(output_shape), 4)
      self.assertEqual(output_shape[0], guide_shape[0])
      self.assertEqual(output_shape[1], guide_shape[1])
      self.assertEqual(output_shape[2], guide_shape[2])
      self.assertEqual(output_shape[3],3)
      self.assertEqual(output_shape_no_offset[3],4)

  def test_interpolate(self):
    pass

  def test_grid_gradient(self):
    for dev in ['/gpu:0']:
      batch_size = 3
      h = 8
      w = 5
      gh = 6
      gw = 3
      d = 7
      i_chans = 3
      o_chans = 3
      grid_shape = [batch_size, gh, gw, d, (1+i_chans)*o_chans]
      guide_shape = [batch_size, h, w]
      input_shape = [batch_size, h, w, i_chans]
      output_shape = [batch_size, h, w, o_chans]

      grid_data = np.random.rand(*grid_shape).astype(np.float32)
      guide_data = 0.8*np.random.rand(*guide_shape).astype(np.float32)+0.1
      input_data = np.random.rand(*input_shape).astype(np.float32)

      with tf.device(dev):
        grid_tensor = tf.convert_to_tensor(grid_data,
                                           name='data',
                                           dtype=tf.float32)
        guide_tensor = tf.convert_to_tensor(guide_data,
                                            name='guide',
                                            dtype=tf.float32)
        input_tensor = tf.convert_to_tensor(input_data,
                                            name='input',
                                            dtype=tf.float32)

        output_tensor = ops.bilateral_slice_apply(grid_tensor, guide_tensor, input_tensor, has_offset=True)

      with self.test_session():
        err = tf.test.compute_gradient_error(
            grid_tensor,
            grid_shape,
            output_tensor,
            output_shape)

        self.assertLess(err, 3e-4)

  def test_guide_gradient(self):
    #TODO: this does not work yet, differentiable 'max' in the tent: max(1-abs(x), 0)?
    for dev in ['/gpu:0']:
      batch_size = 1
      h = 6
      w = 15
      gh = 3
      gw = 9
      d = 7
      i_chans = 1
      o_chans = 1
      grid_shape = [batch_size, gh, gw, d, (i_chans+1)*o_chans]
      guide_shape = [batch_size, h, w]
      input_shape = [batch_size, h, w, i_chans]
      output_shape = [batch_size, h, w, o_chans]

      grid_data = np.random.rand(*grid_shape).astype(np.float32)
      guide_data = np.random.rand(*guide_shape).astype(np.float32)
      input_data = np.random.rand(*input_shape).astype(np.float32)

      with tf.device(dev):
        grid_tensor = tf.convert_to_tensor(grid_data,
                                           name='data',
                                           dtype=tf.float32)
        guide_tensor = tf.convert_to_tensor(guide_data,
                                            name='guide',
                                            dtype=tf.float32)
        input_tensor = tf.convert_to_tensor(input_data,
                                            name='input',
                                            dtype=tf.float32)

        output_tensor = ops.bilateral_slice_apply(grid_tensor, guide_tensor, input_tensor, has_offset=True)

      with self.test_session():
        num, th = tf.test.compute_gradient(
            guide_tensor,
            guide_shape,
            output_tensor,
            output_shape)


        margin = 1e-2
        idx = np.where(np.abs(th-num) >= margin)

        for i in range(len(idx[0])):
          guide_idx = np.unravel_index(idx[0][i], guide_shape)
          output_idx = np.unravel_index(idx[1][i], output_shape)

        err = tf.test.compute_gradient_error(
            guide_tensor,
            guide_shape,
            output_tensor,
            output_shape)
        self.assertLess(err, margin)


  def test_input_gradient(self):
    for dev in ['/gpu:0']:
      batch_size = 1
      h = 8
      w = 5
      gh = 6
      gw = 3
      d = 7
      i_chans = 3
      o_chans = 3
      grid_shape = [batch_size, gh, gw, d, (1+i_chans)*o_chans]
      guide_shape = [batch_size, h, w]
      input_shape = [batch_size, h, w, i_chans]
      output_shape = [batch_size, h, w, o_chans]

      grid_data = np.random.rand(*grid_shape).astype(np.float32)
      guide_data = np.random.rand(*guide_shape).astype(np.float32)
      input_data = np.random.rand(*input_shape).astype(np.float32)

      with tf.device(dev):
        grid_tensor = tf.convert_to_tensor(grid_data,
                                           name='data',
                                           dtype=tf.float32)
        guide_tensor = tf.convert_to_tensor(guide_data,
                                            name='guide',
                                            dtype=tf.float32)
        input_tensor = tf.convert_to_tensor(input_data,
                                            name='input',
                                            dtype=tf.float32)

        output_tensor = ops.bilateral_slice_apply(grid_tensor, guide_tensor, input_tensor, has_offset=True)

      with self.test_session():
        err = tf.test.compute_gradient_error(
            input_tensor,
            input_shape,
            output_tensor,
            output_shape)

        self.assertLess(err, 3e-4)
