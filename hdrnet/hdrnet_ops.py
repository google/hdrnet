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

"""Python interface to custom Tensorflow operations for HDRnet."""

import os
import tensorflow as tf
from tensorflow.python.framework import ops

__all__ = ['bilateral_slice']

path = os.path.dirname(os.path.abspath(__file__))
path = tf.resource_loader.get_path_to_datafile(
    os.path.join(path, 'lib', 'hdrnet_ops.so'))

_hdrnet = tf.load_op_library(path)

# -- Register operations ------------------------------------------------------
bilateral_slice = _hdrnet.bilateral_slice
bilateral_slice_apply = _hdrnet.bilateral_slice_apply

# ----------- Register gradients ----------------------------------------------
@ops.RegisterGradient('BilateralSlice')
def _bilateral_slice_grad(op, grad):
  grid_tensor = op.inputs[0]
  guide_tensor = op.inputs[1]
  return _hdrnet.bilateral_slice_grad(grid_tensor, guide_tensor, grad)


@ops.RegisterGradient('BilateralSliceApply')
def _bilateral_slice_grad(op, grad):
  grid_tensor = op.inputs[0]
  guide_tensor = op.inputs[1]
  input_tensor = op.inputs[2]
  has_offset = op.get_attr('has_offset')
  return _hdrnet.bilateral_slice_apply_grad(
      grid_tensor, guide_tensor, input_tensor, grad, has_offset=has_offset) 


# ----------- Register Shape inference ----------------------------------------
@ops.RegisterShape('BilateralSlice')
def _bilateral_slice_shape(op):
  input_tensor = op.inputs[0]
  guide_tensor = op.inputs[1]
  return [guide_tensor.get_shape().concatenate(input_tensor.get_shape()[-1])]


@ops.RegisterShape('BilateralSliceApply')
def _bilateral_slice_shape(op):
  grid_tensor = op.inputs[0]
  guide_tensor = op.inputs[1]
  input_tensor = op.inputs[2]

  has_offset = op.get_attr('has_offset')
  chan_in = input_tensor.get_shape()[-1]
  chan_grid = grid_tensor.get_shape()[-1]

  if has_offset:
    chan_out = chan_grid // (chan_in+1)
  else:
    chan_out = chan_grid // chan_in
  return [guide_tensor.get_shape().concatenate(chan_out)]
