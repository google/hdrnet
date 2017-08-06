#!/usr/bin/env python
# encoding: utf-8
# -------------------------------------------------------------------
# File:    train.py
# Author:  Michael Gharbi <gharbi@mit.edu>
# Created: 2016-10-25
# -------------------------------------------------------------------
# 
# 
# 
# ------------------------------------------------------------------#
"""Train a model."""

import argparse
import os
import time
import re

import numpy as np

import caffe
import tensorflow as tf

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2

import hdrnet.hdrnet_ops as hops

transfer = {
  "conv1/weights:0": "inference/coefficients/splat/conv1/weights:0",
  "conv1/biases:0": "inference/coefficients/splat/conv1/biases:0",
  "conv2/weights:0": "inference/coefficients/splat/conv2/weights:0",
  "conv2/biases:0": "inference/coefficients/splat/conv2/BatchNorm/beta:0",
  "conv3/weights:0": "inference/coefficients/splat/conv3/weights:0",
  "conv3/biases:0": "inference/coefficients/splat/conv3/BatchNorm/beta:0",
  "conv4/weights:0": "inference/coefficients/splat/conv4/weights:0",
  "conv4/biases:0": "inference/coefficients/splat/conv4/BatchNorm/beta:0",

  "global_conv1/weights:0": "inference/coefficients/global/conv1/weights:0",
  "global_conv1/biases:0": "inference/coefficients/global/conv1/BatchNorm/beta:0",
  "global_conv2/weights:0": "inference/coefficients/global/conv2/weights:0",
  "global_conv2/biases:0": "inference/coefficients/global/conv2/BatchNorm/beta:0",
  "global_fc1/weights:0": "inference/coefficients/global/fc1/weights:0",
  "global_fc1/biases:0": "inference/coefficients/global/fc1/BatchNorm/beta:0",
  "global_fc2/weights:0": "inference/coefficients/global/fc2/weights:0",
  "global_fc2/biases:0": "inference/coefficients/global/fc2/BatchNorm/beta:0",
  "global_fc3/weights:0": "inference/coefficients/global/fc3/weights:0",

  "grid_conv1/weights:0": "inference/coefficients/local/conv1/weights:0",
  "grid_conv1/biases:0": "inference/coefficients/local/conv1/BatchNorm/beta:0",

  "grid_conv2/weights:0": "inference/coefficients/local/conv2/weights:0",

  "post_fusion_conv/weights:0": "inference/coefficients/prediction/conv1/weights:0",
  "post_fusion_conv/biases:0": "inference/coefficients/prediction/conv1/biases:0",
  "guide/guide/ccm:0": "inference/guide/ccm:0",
  "guide/guide/ccm_bias:0": "inference/guide/ccm_bias:0",
  "guide/shifts:0": "inference/guide/shifts:0",
  "guide/slopes:0": "inference/guide/slopes:0",
  "guide/channel_mixing/weights:0": "inference/guide/channel_mixing/weights:0",
  "guide/channel_mixing/biases:0": "inference/guide/channel_mixing/biases:0",
}

transfer_sum = {
  "grid_conv2/biases:0": "inference/coefficients/global/fc3/biases:0",
  "global_fc3/biases:0": "inference/coefficients/global/fc3/biases:0",
}

test_tensors = {
  'inference/conv1/Relu:0': 'train/inference/coefficients/splat/conv1/Relu:0',
  'inference/conv2/Relu:0': 'train/inference/coefficients/splat/conv2/Relu:0',
  'inference/conv3/Relu:0': 'train/inference/coefficients/splat/conv3/Relu:0',
  'inference/conv4/Relu:0': 'train/inference/coefficients/splat/conv4/Relu:0',
  'inference/global_conv1/Relu:0': 'train/inference/coefficients/global/conv1/Relu:0',
  'inference/global_conv2/Relu:0': 'train/inference/coefficients/global/conv2/Relu:0',
  'inference/global_fc1/Relu:0': 'train/inference/coefficients/global/fc1/Relu:0',
  'inference/global_fc2/Relu:0': 'train/inference/coefficients/global/fc2/Relu:0',
  'inference/global_fc3/MatMul:0': 'train/inference/coefficients/global/fc3/MatMul:0',
  'inference/grid_conv1/Relu:0': 'train/inference/coefficients/local/conv1/Relu:0',
  'inference/grid_conv2/Conv2D:0': 'train/inference/coefficients/local/conv2/convolution:0',
  'inference/fusion/Relu:0': 'train/inference/coefficients/fusion/Relu:0',
  'inference/affine_coefficients:0': 'train/inference/coefficients/prediction/unroll_grid/stack_1:0',
  'inference/affine_coefficients:0': 'train/inference/coefficients/prediction/unroll_grid/stack_1:0',
  # 'inference/guide/Sum:0': 'train/inference/guide/curve/Sum:0',
  # 'inference/guide/channel_mixing/BiasAdd:0': 'train/inference/guide/channel_mixing/BiasAdd:0',
  # 'inference/guide/guide/Reshape_1:0': 'train/inference/guide/ccm/Reshape_1:0',
}


def main(args):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  if not args.batch_norm:
    for t in transfer:
      if "BatchNorm/beta" in transfer[t]:
        transfer[t] = transfer[t].replace("BatchNorm/beta", "biases")
        print t, transfer[t]

  # Prepare a random input image
  test_data = np.random.rand(256, 256, 3).astype(np.float32)

  src_path = tf.train.latest_checkpoint(args.src)
  if src_path is None:
    raise ValueError ('Transplant: could not find a checkpoint in {}'.format(args.src))
  # Load src_path and metagraph, list all vars
  metapath = ".".join([src_path, "meta"])
  saver = tf.train.import_meta_graph(metapath)
  input_t = tf.get_default_graph().get_tensor_by_name('DataPipeline/shuffle_batch:0')
  bs_src = input_t.get_shape().as_list()[0]
  input_batch_src = np.zeros((bs_src, 256, 256, 3), dtype=np.float32)
  input_batch_src[0, :, :, :] = test_data[...]

  test_src = {}
  for k in test_tensors:
    test_src[k] = tf.get_default_graph().get_tensor_by_name(k)

  with tf.Session(config=config) as sess:
    saver.restore(sess, src_path)

    reject = re.compile(r".*(Adam|global_step|beta1_power|beta2_power|learning_rate|ExponentialMovingAverage|moving_mean|moving_variance).*")
    overrides = {}
    print "Fetching source variables tensors"
    for v in tf.global_variables():
      if reject.match(v.name):
        continue
      value = sess.run(v)

      if v.name in transfer.keys():
        # Take into account updated guide implementation
        if "channel_mixing/biases" in v.name:
          overrides[transfer[v.name]] = value + 0.5/8.0
          print "channel mix", value + 0.5/8.0, value
        elif "channel_mixing/weights" in v.name:
          overrides[transfer[v.name]] = value*(8.0-1.0)/8.0
          print "channel mix", value*(8.0-1.0)/8.0, value
        else:
          overrides[transfer[v.name]] = value
      elif v.name in transfer_sum.keys():
        dst = transfer_sum[v.name]
        if dst in overrides.keys():
          overrides[dst] += value
        else:
          overrides[dst] = value

    # print "Evaluating test tensors for the source"
    # test_result_src = sess.run(test_src, feed_dict={input_t: input_batch_src})

  tf.reset_default_graph()
  # Load dst_path and metagraph, list all vars
  dst_path = tf.train.latest_checkpoint(args.dst)
  if dst_path is None:
    raise ValueError ('Transplant: could not find a checkpoint in {}'.format(args.dst))
  metapath = ".".join([dst_path, "meta"])
  saver = tf.train.import_meta_graph(metapath)
  dst_saver = tf.train.Saver(tf.global_variables())

  input_t = tf.get_default_graph().get_tensor_by_name('train_data/shuffle_batch:2')
  input_t_fr = tf.get_default_graph().get_tensor_by_name('train_data/shuffle_batch:0')
  bs_dst = input_t.get_shape().as_list()[0]
  input_batch_dst = np.zeros((bs_dst, 256, 256, 3), dtype=np.float32)
  input_batch_dst[0, :, :, :] = test_data[...]

  test_dst = {}
  for k in test_tensors:
    test_dst[k] = tf.get_default_graph().get_tensor_by_name(test_tensors[k])

  with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()

    print "Overriding target tensors"
    for v in overrides:
      print v
      var = tf.get_default_graph().get_tensor_by_name(v)
      assign_op = tf.assign(var, overrides[v])
      sess.run(assign_op)

    print "Batch norm and moving averages:"
    for v in tf.global_variables():
      if 'moving_variance' in v.name or 'moving_mean' in v.name or 'BatchNorm/beta' in v.name:
        var = tf.get_default_graph().get_tensor_by_name(v.name)
        var = sess.run(var)
        print v.name, np.mean(np.ravel(var)), np.std(np.ravel(var))
    print "Batch norm constants:"

    feed_dict = {}
    # for v in tf.get_default_graph().as_graph_def().node:
    #   if 'BatchNorm/batchnorm/add/y' in v.name:
    #     print "set to 0", v.name
    #     var = tf.get_default_graph().get_tensor_by_name(v.name+":0")
    #     feed_dict[var] = 0.0

    print "Evaluating test tensors for the destination"
    feed_dict[input_t] = input_batch_dst
    feed_dict[input_t_fr] = input_batch_dst
    # test_result_dst = sess.run(test_dst, feed_dict=feed_dict)
    # for k in test_result_src:
    #   s_ = test_result_src[k][0, ...]
    #   t_ = test_result_dst[k][0, ...]
    #   diff = np.amax(np.abs(s_-t_))
    #   print k, s_.shape, t_.shape, "error:", diff
    #   assert (diff < 1e-5)

    new_path = os.path.join(os.path.dirname(dst_path), 'upgraded_from_sig2017.ckpt')
    print "Saving checkpoint:", new_path
    if not os.path.exists(args.dst):
      os.makedirs(args.dst)
    dst_saver.save(sess, new_path, global_step=0)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--src', type=str, required=True)
  parser.add_argument('--dst', type=str, required=True)
  parser.add_argument('--batch_norm', dest='batch_norm', action='store_true', help='normalize batches. If False, uses the moving averages')
  parser.add_argument('--nobatch_norm', dest='batch_norm', action='store_false')
  parser.set_defaults(batch_norm=False)
  args = parser.parse_args()
  main(args)
