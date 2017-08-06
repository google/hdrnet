#!/usr/bin/env python
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

"""Train a model."""

import argparse
import logging
import numpy as np
import os
import setproctitle
import tensorflow as tf
import time

import hdrnet.metrics as metrics

import hdrnet.models as models
import hdrnet.data_pipeline as dp


logging.basicConfig(format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
log = logging.getLogger("train")
log.setLevel(logging.INFO)


def log_hook(sess, log_fetches):
  """Message display at every log step."""
  data = sess.run(log_fetches)
  step = data['step']
  loss = data['loss']
  psnr = data['psnr']
  log.info('Step {} | loss = {:.4f} | psnr = {:.1f} dB'.format(step, loss, psnr))


def main(args, model_params, data_params):
  procname = os.path.basename(args.checkpoint_dir)
  setproctitle.setproctitle('hdrnet_{}'.format(procname))

  log.info('Preparing summary and checkpoint directory {}'.format(
      args.checkpoint_dir))
  if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

  tf.set_random_seed(1234)  # Make experiments repeatable

  # Select an architecture
  mdl = getattr(models, args.model_name)

  # Add model parameters to the graph (so they are saved to disk at checkpoint)
  for p in model_params:
    p_ = tf.convert_to_tensor(model_params[p], name=p)
    tf.add_to_collection('model_params', p_)

  # --- Train/Test datasets ---------------------------------------------------
  data_pipe = getattr(dp, args.data_pipeline)
  with tf.variable_scope('train_data'):
    train_data_pipeline = data_pipe(
        args.data_dir,
        shuffle=True,
        batch_size=args.batch_size, nthreads=args.data_threads,
        fliplr=args.fliplr, flipud=args.flipud, rotate=args.rotate,
        random_crop=args.random_crop, params=data_params,
        output_resolution=args.output_resolution)
    train_samples = train_data_pipeline.samples

  if args.eval_data_dir is not None:
    with tf.variable_scope('eval_data'):
      eval_data_pipeline = data_pipe(
          args.eval_data_dir,
          shuffle=False,
          batch_size=1, nthreads=1,
          fliplr=False, flipud=False, rotate=False,
          random_crop=False, params=data_params,
          output_resolution=args.output_resolution)
      eval_samples = train_data_pipeline.samples
  # ---------------------------------------------------------------------------

  # Training graph
  with tf.name_scope('train'):
    with tf.variable_scope('inference'):
      prediction = mdl.inference(
          train_samples['lowres_input'], train_samples['image_input'],
          model_params, is_training=True)
    loss = metrics.l2_loss(train_samples['image_output'], prediction)
    psnr = metrics.psnr(train_samples['image_output'], prediction)

  # Evaluation graph
  if args.eval_data_dir is not None:
    with tf.name_scope('eval'):
      with tf.variable_scope('inference', reuse=True):
        eval_prediction = mdl.inference(
            eval_samples['lowres_input'], eval_samples['image_input'],
            model_params, is_training=False)
      eval_psnr = metrics.psnr(eval_samples['image_output'], prediction)

  # Optimizer
  global_step = tf.contrib.framework.get_or_create_global_step()
  with tf.name_scope('optimizer'):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    updates = tf.group(*update_ops, name='update_ops')
    log.info("Adding {} update ops".format(len(update_ops)))
    with tf.control_dependencies([updates]):
      opt = tf.train.AdamOptimizer(args.learning_rate)
      minimize = opt.minimize(loss, name='optimizer', global_step=global_step)

  # Average loss and psnr for display
  with tf.name_scope("moving_averages"):
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_ma = ema.apply([loss, psnr])
    loss = ema.average(loss)
    psnr = ema.average(psnr)

  # Training stepper operation
  train_op = tf.group(minimize, update_ma)

  # Save a few graphs to tensorboard
  summaries = [
    tf.summary.scalar('loss', loss),
    tf.summary.scalar('psnr', psnr),
    tf.summary.scalar('learning_rate', args.learning_rate),
    tf.summary.scalar('batch_size', args.batch_size),
  ]

  log_fetches = {
      "step": global_step,
      "loss": loss,
      "psnr": psnr}

  # Train config
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True  # Do not canibalize the entire GPU
  sv = tf.train.Supervisor(
      logdir=args.checkpoint_dir,
      save_summaries_secs=args.summary_interval,
      save_model_secs=args.checkpoint_interval)

  # Train loop
  with sv.managed_session(config=config) as sess:
    sv.loop(args.log_interval, log_hook, (sess, log_fetches))
    last_eval = time.time()
    while True:
      if sv.should_stop():
        log.info("stopping supervisor")
        break
      try:
        step, _ = sess.run([global_step, train_op])
        since_eval = time.time()-last_eval

        if args.eval_data_dir is not None and since_eval > args.eval_interval:
          log.info("Evaluating on {} images at step {}".format(
              eval_data_pipeline.nsamples, step))

          p_ = 0
          for it in range(eval_data_pipeline.nsamples):
            p_ += sess.run(eval_psnr)
          p_ /= eval_data_pipeline.nsamples

          sv.summary_writer.add_summary(tf.Summary(value=[
            tf.Summary.Value(tag="psnr/eval", simple_value=p_)]), global_step=step)

          log.info("  Evaluation PSNR = {:.1f} dB".format(p_))

          last_eval = time.time()

      except tf.errors.AbortedError:
        log.error("Aborted")
        break
      except KeyboardInterrupt:
        break
    chkpt_path = os.path.join(args.checkpoint_dir, 'on_stop.ckpt')
    log.info("Training complete, saving chkpt {}".format(chkpt_path))
    sv.saver.save(sess, chkpt_path)
    sv.request_stop()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # pylint: disable=line-too-long
  # ----------------------------------------------------------------------------
  req_grp = parser.add_argument_group('required')
  req_grp.add_argument('checkpoint_dir', default=None, help='directory to save checkpoints to.')
  req_grp.add_argument('data_dir', default=None, help='input directory containing the training .tfrecords or images.')
  req_grp.add_argument('--eval_data_dir', default=None, type=str, help='directory with the validation data.')

  # Training, logging and checkpointing parameters
  train_grp = parser.add_argument_group('training')
  train_grp.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate for the stochastic gradient update.')
  train_grp.add_argument('--log_interval', type=int, default=1, help='interval between log messages (in s).')
  train_grp.add_argument('--summary_interval', type=int, default=120, help='interval between tensorboard summaries (in s)')
  train_grp.add_argument('--checkpoint_interval', type=int, default=600, help='interval between model checkpoints (in s)')
  train_grp.add_argument('--eval_interval', type=int, default=3600, help='interval between evaluations (in s)')

  # Debug and perf profiling
  debug_grp = parser.add_argument_group('debug and profiling')
  debug_grp.add_argument('--profiling', dest='profiling', action='store_true', help='outputs a profiling trace.')
  debug_grp.add_argument('--noprofiling', dest='profiling', action='store_false')

  # Data pipeline and data augmentation
  data_grp = parser.add_argument_group('data pipeline')
  data_grp.add_argument('--batch_size', default=16, type=int, help='size of a batch for each gradient update.')
  data_grp.add_argument('--data_threads', default=2, help='number of threads to load and enqueue samples.')
  data_grp.add_argument('--rotate', dest="rotate", action="store_true", help='rotate data augmentation.')
  data_grp.add_argument('--norotate', dest="rotate", action="store_false")
  data_grp.add_argument('--flipud', dest="flipud", action="store_true", help='flip up/down data augmentation.')
  data_grp.add_argument('--noflipud', dest="flipud", action="store_false")
  data_grp.add_argument('--fliplr', dest="fliplr", action="store_true", help='flip left/right data augmentation.')
  data_grp.add_argument('--nofliplr', dest="fliplr", action="store_false")
  data_grp.add_argument('--random_crop', dest="random_crop", action="store_true", help='random crop data augmentation.')
  data_grp.add_argument('--norandom_crop', dest="random_crop", action="store_false")

  # Model parameters
  model_grp = parser.add_argument_group('model_params')
  model_grp.add_argument('--model_name', default=models.__all__[0], type=str, help='classname of the model to use.', choices=models.__all__)
  model_grp.add_argument('--data_pipeline', default='ImageFilesDataPipeline', help='classname of the data pipeline to use.', choices=dp.__all__)
  model_grp.add_argument('--net_input_size', default=256, type=int, help="size of the network's lowres image input.")
  model_grp.add_argument('--output_resolution', default=[512, 512], type=int, nargs=2, help='resolution of the output image.')
  model_grp.add_argument('--batch_norm', dest='batch_norm', action='store_true', help='normalize batches. If False, uses the moving averages.')
  model_grp.add_argument('--nobatch_norm', dest='batch_norm', action='store_false')
  model_grp.add_argument('--channel_multiplier', default=1, type=int,  help='Factor to control net throughput (number of intermediate channels).')
  model_grp.add_argument('--guide_complexity', default=16, type=int,  help='Control complexity of the guide network.')

  # Bilateral grid parameters
  model_grp.add_argument('--luma_bins', default=8, type=int,  help='Number of BGU bins for the luminance.')
  model_grp.add_argument('--spatial_bin', default=16, type=int,  help='Size of the spatial BGU bins (pixels).')

  parser.set_defaults(
      profiling=False,
      flipud=False,
      fliplr=False,
      rotate=False,
      random_crop=True,
      batch_norm=False)
  # ----------------------------------------------------------------------------
  # pylint: enable=line-too-long

  args = parser.parse_args()

  model_params = {}
  for a in model_grp._group_actions:
    model_params[a.dest] = getattr(args, a.dest, None)

  data_params = {}
  for a in data_grp._group_actions:
    data_params[a.dest] = getattr(args, a.dest, None)

  main(args, model_params, data_params)
