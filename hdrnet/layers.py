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

"""Shortcuts for some graph operators."""

import tensorflow as tf
import numpy as np

from hdrnet import hdrnet_ops

w_initializer = tf.contrib.layers.variance_scaling_initializer
b_initializer = tf.constant_initializer

def conv(inputs, num_outputs, kernel_size, stride=1, rate=1,
    use_bias=True,
    batch_norm=False, is_training=False,
    activation_fn=tf.nn.relu, 
    scope=None, reuse=False):
  if batch_norm:
    normalizer_fn = tf.contrib.layers.batch_norm
    b_init = None
  else:
    normalizer_fn = None
    if use_bias:
      b_init = b_initializer(0.0)
    else:
      b_init = None

  output = tf.contrib.layers.convolution2d(
      inputs=inputs,
      num_outputs=num_outputs, kernel_size=kernel_size, 
      stride=stride, padding='SAME',
      rate=rate,
      weights_initializer=w_initializer(),
      biases_initializer=b_init,
      normalizer_fn=normalizer_fn,
      normalizer_params={
        'center':True, 'is_training':is_training,
        'variables_collections':{
          'beta':[tf.GraphKeys.BIASES],
          'moving_mean':[tf.GraphKeys.MOVING_AVERAGE_VARIABLES],
          'moving_variance':[tf.GraphKeys.MOVING_AVERAGE_VARIABLES]},
        }, 
      activation_fn=activation_fn, 
      variables_collections={'weights':[tf.GraphKeys.WEIGHTS], 'biases':[tf.GraphKeys.BIASES]},
      outputs_collections=[tf.GraphKeys.ACTIVATIONS],
      scope=scope, reuse=reuse)
  return output


def fc(inputs, num_outputs,
    use_bias=True,
    batch_norm=False, is_training=False,
    activation_fn=tf.nn.relu, 
    scope=None):
  if batch_norm:
    normalizer_fn = tf.contrib.layers.batch_norm
    b_init = None
  else:
    normalizer_fn = None
    if use_bias:
      b_init = b_initializer(0.0)
    else:
      b_init = None

  output = tf.contrib.layers.fully_connected(
      inputs=inputs,
      num_outputs=num_outputs,
      weights_initializer=w_initializer(),
      biases_initializer=b_init,
      normalizer_fn=normalizer_fn,
      normalizer_params={
        'center':True, 'is_training':is_training,
        'variables_collections':{
          'beta':[tf.GraphKeys.BIASES],
          'moving_mean':[tf.GraphKeys.MOVING_AVERAGE_VARIABLES],
          'moving_variance':[tf.GraphKeys.MOVING_AVERAGE_VARIABLES]},
        }, 
      activation_fn=activation_fn, 
      variables_collections={'weights':[tf.GraphKeys.WEIGHTS], 'biases':[tf.GraphKeys.BIASES]},
      scope=scope)
  return output


# -----------------------------------------------------------------------------

# pylint: disable=redefined-builtin
def bilateral_slice(grid, guide, name=None):
  """Slices into a bilateral grid using the guide map.

  Args:
    grid: (Tensor) [batch_size, grid_h, grid_w, depth, n_outputs]
      grid to slice from.
    guide: (Tensor) [batch_size, h, w ] guide map to slice along.
    name: (string) name for the operation.
  Returns:
    sliced: (Tensor) [batch_size, h, w, n_outputs] sliced output.
  """

  with tf.name_scope(name):
    gridshape = grid.get_shape().as_list()
    if len(gridshape) == 6:
      _, _, _, _, n_out, n_in = gridshape
      grid = tf.concat(tf.unstack(grid, None, axis=5), 4)

    sliced = hdrnet_ops.bilateral_slice(grid, guide)

    if len(gridshape) == 6:
      sliced = tf.stack(tf.split(sliced, n_in, axis=3), axis=4)
    return sliced
# pylint: enable=redefined-builtin


def bilateral_slice_apply(grid, guide, input_image, has_offset=True, name=None):
  """Slices into a bilateral grid using the guide map.

  Args:
    grid: (Tensor) [batch_size, grid_h, grid_w, depth, n_outputs]
      grid to slice from.
    guide: (Tensor) [batch_size, h, w ] guide map to slice along.
    input_image: (Tensor) [batch_size, h, w, n_input] input data onto which to
      apply the affine transform.
    name: (string) name for the operation.
  Returns:
    sliced: (Tensor) [batch_size, h, w, n_outputs] sliced output.
  """

  with tf.name_scope(name):
    gridshape = grid.get_shape().as_list()
    if len(gridshape) == 6:
      gs = tf.shape(grid)
      _, _, _, _, n_out, n_in = gridshape
      grid = tf.reshape(grid, tf.stack([gs[0], gs[1], gs[2], gs[3], gs[4]*gs[5]]))
      # grid = tf.concat(tf.unstack(grid, None, axis=5), 4)

    sliced = hdrnet_ops.bilateral_slice_apply(grid, guide, input_image, has_offset=has_offset)
    return sliced
# pylint: enable=redefined-builtin


# pylint: disable=redefined-builtin
def apply(sliced, input_image, has_affine_term=True, name=None):
  """Applies a sliced affined model to the input image.

  Args:
    sliced: (Tensor) [batch_size, h, w, n_output, n_input+1] affine coefficients
    input_image: (Tensor) [batch_size, h, w, n_input] input data onto which to
      apply the affine transform.
    name: (string) name for the operation.
  Returns:
    ret: (Tensor) [batch_size, h, w, n_output] the transformed data.
  Raises:
    ValueError: if the input is not properly dimensioned.
    ValueError: if the affine model parameter dimensions do not match the input.
  """

  with tf.name_scope(name):
    if len(input_image.get_shape().as_list()) != 4:
      raise ValueError('input image should have dims [b,h,w,n_in].')
    in_shape = input_image.get_shape().as_list()
    sliced_shape = sliced.get_shape().as_list()
    if (in_shape[:-1] != sliced_shape[:-2]):
      raise ValueError('input image and affine coefficients'
                       ' dimensions do not match: {} and {}'.format(
                       in_shape, sliced_shape))
    _, _, _, n_out, n_in = sliced.get_shape().as_list()
    if has_affine_term:
      n_in -= 1

    scale = sliced[:, :, :, :, :n_in]

    if has_affine_term:
      offset = sliced[:, :, :, :, n_in]

    out_channels = []
    for chan in range(n_out):
      ret = scale[:, :, :, chan, 0]*input_image[:, :, :, 0]
      for chan_i in range(1, n_in):
        ret += scale[:, :, :, chan, chan_i]*input_image[:, :, :, chan_i]
      if has_affine_term:
        ret += offset[:, :, :, chan]
      ret = tf.expand_dims(ret, 3)
      out_channels.append(ret)

    ret = tf.concat(out_channels, 3)

  return ret
# pylint: enable=redefined-builtin
