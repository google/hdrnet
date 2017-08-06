#!/usr/bin/env python
# encoding: utf-8
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

"""Freeze graph weights; use to optimize runtime."""

import argparse
import logging
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.core.framework import graph_pb2

import hdrnet.utils as utils
import hdrnet.models as models


logging.basicConfig(format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
log = logging.getLogger("train")
log.setLevel(logging.INFO)


def save(data, filepath):
  log.info("Saving {}".format(filepath))
  with open(filepath, 'wb') as fid:
    fid.write(data.tobytes())


def main(args):
  # Read model parameters
  checkpoint_path = tf.train.latest_checkpoint(args.checkpoint_dir)
  if checkpoint_path is None:
    log.error('Could not find a checkpoint in {}'.format(args.checkpoint_dir))
    return
  metapath = ".".join([checkpoint_path, "meta"])
  log.info("Loading {}".format(metapath))
  tf.train.import_meta_graph(metapath)
  with tf.Session() as sess:
    model_params = utils.get_model_params(sess)

  if not hasattr(models, model_params['model_name']):
    log.error("Model {} does not exist".format(model_params['model_name']))
    return
  mdl = getattr(models, model_params['model_name'])

  # Instantiate new evaluation graph
  tf.reset_default_graph()
  sz = model_params['net_input_size']

  log.info("Model {}".format(model_params['model_name']))

  input_tensor = tf.placeholder(tf.float32, [1, sz, sz, 3], name='lowres_input')
  with tf.variable_scope('inference'):
    prediction = mdl.inference(input_tensor, input_tensor, model_params, is_training=False)
  if model_params["model_name" ] == "HDRNetGaussianPyrNN":
    output_tensor = tf.get_collection('packed_coefficients')[0]
    output_tensor = tf.transpose(tf.squeeze(output_tensor), [3, 2, 0, 1, 4], name="output_coefficients")
    log.info("Output shape".format(output_tensor.get_shape()))
  else:
    output_tensor = tf.get_collection('packed_coefficients')[0]
    output_tensor = tf.transpose(tf.squeeze(output_tensor), [3, 2, 0, 1, 4], name="output_coefficients")
    log.info("Output shape {}".format(output_tensor.get_shape()))
  saver = tf.train.Saver()

  gdef = tf.get_default_graph().as_graph_def()

  log.info("Restoring weights from {}".format(checkpoint_path))
  test_graph_name = "test_graph.pbtxt"
  with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    tf.train.write_graph(sess.graph, args.checkpoint_dir, test_graph_name)

    input_graph_path = os.path.join(args.checkpoint_dir, test_graph_name)
    output_graph_path = os.path.join(args.checkpoint_dir, "frozen_graph.pb")
    input_saver_def_path = ""
    input_binary = False
    output_binary = True
    input_node_names = input_tensor.name.split(":")[0]
    output_node_names = output_tensor.name.split(":")[0]
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    clear_devices = False

    log.info("Freezing to {}".format(output_graph_path))
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_graph_path, clear_devices, "")
    log.info('input tensor: {} {}'.format(input_tensor.name, input_tensor.shape))
    log.info('output tensor: {} {}'.format(output_tensor.name, output_tensor.shape))

    # Dump guide parameters
    if model_params['model_name'] == 'HDRNetCurves':
      g = tf.get_default_graph()
      ccm = g.get_tensor_by_name('inference/guide/ccm:0')
      ccm_bias = g.get_tensor_by_name('inference/guide/ccm_bias:0')
      shifts = g.get_tensor_by_name('inference/guide/shifts:0')
      slopes = g.get_tensor_by_name('inference/guide/slopes:0')
      mixing_weights = g.get_tensor_by_name('inference/guide/channel_mixing/weights:0')
      mixing_bias = g.get_tensor_by_name('inference/guide/channel_mixing/biases:0')

      ccm_, ccm_bias_, shifts_, slopes_, mixing_weights_, mixing_bias_ = sess.run(
              [ccm, ccm_bias, shifts, slopes, mixing_weights, mixing_bias])
      shifts_ = np.squeeze(shifts_).astype(np.float32)
      slopes_ = np.squeeze(slopes_).astype(np.float32)
      mix_matrix_dump = np.append(np.squeeze(mixing_weights_), mixing_bias_[0]).astype(np.float32)
      ccm34_ = np.vstack((ccm_, ccm_bias_[np.newaxis, :]))

      save(ccm34_.T, os.path.join(args.checkpoint_dir, 'guide_ccm_f32_3x4.bin'))
      save(shifts_.T, os.path.join(args.checkpoint_dir, 'guide_shifts_f32_16x3.bin'))
      save(slopes_.T, os.path.join(args.checkpoint_dir, 'guide_slopes_f32_16x3.bin'))
      save(mix_matrix_dump, os.path.join(args.checkpoint_dir, 'guide_mix_matrix_f32_1x4.bin'))

    elif model_params['model_name'] == "HDRNetGaussianPyrNN":
      g = tf.get_default_graph()
      for lvl in range(3):
        conv1_w = g.get_tensor_by_name('inference/guide/level_{}/conv1/weights:0'.format(lvl))
        conv1_b = g.get_tensor_by_name('inference/guide/level_{}/conv1/BatchNorm/beta:0'.format(lvl))
        conv1_mu = g.get_tensor_by_name('inference/guide/level_{}/conv1/BatchNorm/moving_mean:0'.format(lvl))
        conv1_sigma = g.get_tensor_by_name('inference/guide/level_{}/conv1/BatchNorm/moving_variance:0'.format(lvl))
        conv1_eps = g.get_tensor_by_name('inference/guide/level_{}/conv1/BatchNorm/batchnorm/add/y:0'.format(lvl))
        conv2_w = g.get_tensor_by_name('inference/guide/level_{}/conv2/weights:0'.format(lvl))
        conv2_b = g.get_tensor_by_name('inference/guide/level_{}/conv2/biases:0'.format(lvl))

        conv1w_, conv1b_, conv1mu_, conv1sigma_, conv1eps_, conv2w_, conv2b_ = sess.run(
            [conv1_w, conv1_b, conv1_mu, conv1_sigma, conv1_eps, conv2_w, conv2_b])

        conv1b_ -= conv1mu_/np.sqrt((conv1sigma_+conv1eps_))
        conv1w_ = conv1w_/np.sqrt((conv1sigma_+conv1eps_))

        conv1w_ = np.squeeze(conv1w_.astype(np.float32))
        conv1b_ = np.squeeze(conv1b_.astype(np.float32))
        conv1b_ = conv1b_[np.newaxis, :]

        conv2w_ = np.squeeze(conv2w_.astype(np.float32))
        conv2b_ = np.squeeze(conv2b_.astype(np.float32))

        conv2 = np.append(conv2w_, conv2b_)
        conv1 = np.vstack([conv1w_, conv1b_])

        save(conv1.T, os.path.join(args.checkpoint_dir, 'guide_level{}_conv1.bin'.format(lvl)))
        save(conv2, os.path.join(args.checkpoint_dir, 'guide_level{}_conv2.bin'.format(lvl)))

    elif model_params['model_name'] in "HDRNetPointwiseNNGuide":
      g = tf.get_default_graph()
      conv1_w = g.get_tensor_by_name('inference/guide/conv1/weights:0')
      conv1_b = g.get_tensor_by_name('inference/guide/conv1/BatchNorm/beta:0')
      conv1_mu = g.get_tensor_by_name('inference/guide/conv1/BatchNorm/moving_mean:0')
      conv1_sigma = g.get_tensor_by_name('inference/guide/conv1/BatchNorm/moving_variance:0')
      conv1_eps = g.get_tensor_by_name('inference/guide/conv1/BatchNorm/batchnorm/add/y:0')
      conv2_w = g.get_tensor_by_name('inference/guide/conv2/weights:0')
      conv2_b = g.get_tensor_by_name('inference/guide/conv2/biases:0')

      conv1w_, conv1b_, conv1mu_, conv1sigma_, conv1eps_, conv2w_, conv2b_ = sess.run(
          [conv1_w, conv1_b, conv1_mu, conv1_sigma, conv1_eps, conv2_w, conv2_b])

      conv1b_ -= conv1mu_/np.sqrt((conv1sigma_+conv1eps_))
      conv1w_ = conv1w_/np.sqrt((conv1sigma_+conv1eps_))

      conv1w_ = np.squeeze(conv1w_.astype(np.float32))
      conv1b_ = np.squeeze(conv1b_.astype(np.float32))
      conv1b_ = conv1b_[np.newaxis, :]

      conv2w_ = np.squeeze(conv2w_.astype(np.float32))
      conv2b_ = np.squeeze(conv2b_.astype(np.float32))

      conv2 = np.append(conv2w_, conv2b_)
      conv1 = np.vstack([conv1w_, conv1b_])

      save(conv1.T, os.path.join(args.checkpoint_dir, 'guide_conv1.bin'))
      save(conv2, os.path.join(args.checkpoint_dir, 'guide_conv2.bin'))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_dir', default=None, help='')

  args = parser.parse_args()
  main(args)
