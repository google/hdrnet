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

"""Visualize model's weights and activations. """

import argparse
import logging
import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import skimage


logging.basicConfig(format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
log = logging.getLogger("train")
log.setLevel(logging.INFO)

    
def viz_array(a, s=0.1):
  return np.squeeze(np.uint8(np.clip((a-a.mean())/max(a.std(), 1e-4)*s + 0.5, 0, 1)*255))


def main(args):
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  with tf.Graph().as_default() as graph:
    def get_layer(name):
      return graph.get_tensor_by_name("{}:0".format(name))

    path = os.path.join(args.checkpoint_dir, 'optimized_graph.pb')
    if not os.path.exists(path):
      log.error("{} does not exists, please run optimize_graph.sh first".format(path))
    with open(path) as fid:
      graph_def = tf.GraphDef.FromString(fid.read())

    psize = 256
    t_lowres_input = tf.placeholder(tf.float32, (psize, psize, 3))
    t_input = tf.expand_dims(t_lowres_input, 0)

    if args.input is None:
      log.info("Using random input")
      im0 = (np.random.rand(psize,psize,3)/255.0 + 0.5).astype(np.float32)
    else:
      im0 = skimage.img_as_float(skimage.io.imread(args.input))
      im0 = skimage.transform.resize(im0, (psize, psize))

    tf.import_graph_def(graph_def, {'lowres_input': t_input})

    activations = {}
    weights = {}
    biases = {}
    with tf.Session() as sess:
      ops = sess.graph.get_operations()
      for op in ops:
        name = op.name.replace('/', '_')
        if not 'inference' in op.name:
          continue
        if ('Relu' in op.name or 'BiasAdd' in op.name):
          if 'fc' in op.name:
            continue
          a = tf.squeeze(get_layer(op.name), 0)
          log.info("Activation shape {}".format(a.get_shape()))
          a = tf.pad(a, [[0,0], [0,2], [0,0]])
          a = tf.transpose(a, [0, 2, 1])
          sz = tf.shape(a)
          a = tf.reshape(a, (sz[0], sz[1]*sz[2]))
          activations[name] = a
        elif 'weights' in op.name:
          if 'fc' in op.name:
            continue
          w = get_layer(op.name)
          log.info("Weights shape {}".format(w.get_shape()))
          w = tf.pad(w, [[0,1], [0,1], [0,0], [0,0]])
          w = tf.transpose(w, [3, 0, 2, 1])
          sz = tf.shape(w)
          w = tf.reshape(w, (sz[0]*sz[1], sz[2]*sz[3]))
          weights[name] = w
        elif 'biases' in op.name:
          biases[name] = get_layer(op.name)
        log.info("Operation {}".format(op.name))
      
    with tf.Session() as sess:
      tf.global_variables_initializer().run()

      w_ = sess.run(weights, {t_lowres_input: im0})
      for name in w_.keys():
        im_viz = viz_array(w_[name])
        path = os.path.join(args.output_dir, "weights_{}.png".format(name))
        skimage.io.imsave(path, im_viz)

      a_ = sess.run(activations, {t_lowres_input: im0})
      for name in a_.keys():
        im_viz = viz_array(a_[name])
        path = os.path.join(args.output_dir, "activations_{}.png".format(name))
        skimage.io.imsave(path, im_viz)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint_dir', type=str)
  parser.add_argument('output_dir', type=str)
  parser.add_argument('--input', type=str)
  parser.add_argument('--learning_rate', type=float, default=1)
  args = parser.parse_args()

  main(args)
