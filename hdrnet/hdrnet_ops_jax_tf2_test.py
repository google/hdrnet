# Lint as: python3
"""Tests comparing bilateral slice between JAX and TF2."""

import time

import hdrnet_ops as tf2_ops
import jax
from ..jax import bilateral_slice as jax_ops
import numpy as np
import tensorflow as tf


class HdrnetOpsJaxTf2Test(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    print(f'TF2 executing eagerly: {tf.executing_eagerly()}')
    print(f'JAX devices: {jax.devices()}')
    print(
        f'JAX XLA bridge backend platform: {jax.lib.xla_bridge.get_backend().platform}'
    )

    # Add a batch axis to the JAX version.
    self.jax_batch_slice = jax.jit(
        jax.vmap(jax_ops.bilateral_slice, in_axes=0, out_axes=0))

  def test_bilateral_slice_jax_close_to_tf2(self):

    batch_size = 4
    gh = 16
    gw = 12
    gd = 8
    gc = 2
    h = 640
    w = 480

    grid_shape = (batch_size, gh, gw, gd, gc)
    guide_shape = (batch_size, h, w)
    expected_output_shape = (batch_size, h, w, gc)

    grid = np.random.rand(*grid_shape).astype(np.float32)
    guide = np.random.rand(*guide_shape).astype(np.float32)

    tf2_sliced = tf2_ops.bilateral_slice(grid, guide).numpy()
    jax_sliced = self.jax_batch_slice(grid, guide)

    self.assertTupleEqual(tf2_sliced.shape, expected_output_shape)
    self.assertTupleEqual(jax_sliced.shape, expected_output_shape)
    self.assertAllClose(tf2_sliced, jax_sliced)


class HdrnetOpsBenchmark(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    batch_size = 4
    gh = 16
    gw = 12
    gd = 8
    gc = 2
    h = 1024
    w = 768

    self.burn_iterations = 10
    self.benchmark_iterations = 100

    self.grid_shape = (batch_size, gh, gw, gd, gc)
    self.guide_shape = (batch_size, h, w)

    self.grid = np.random.rand(*self.grid_shape).astype(np.float32)
    self.guide = np.random.rand(*self.guide_shape).astype(np.float32)

    print(f'grid_shape: {self.grid_shape}')
    print(f'guide_shape: {self.guide_shape}')
    print(
        f'burning for {self.burn_iterations} iterations, benchmarking {self.benchmark_iterations} iterations'
    )


def _timeit(f, burn_iterations=10, benchmark_iterations=100):
  # Burn.
  for _ in range(burn_iterations):
    f()
  # Benchmark.
  t0 = time.time()
  for _ in range(benchmark_iterations):
    f()
  t1 = time.time()
  elapsed_sec = t1 - t0
  elapsed_ms = elapsed_sec * 1000
  mean_elapsed_ms = elapsed_ms / benchmark_iterations
  return mean_elapsed_ms, elapsed_ms


class HdrnetOpsJaxBenchmark(HdrnetOpsBenchmark):

  def setUp(self):
    super().setUp()
    print(f'JAX devices: {jax.devices()}')
    print(
        f'JAX XLA bridge backend platform: {jax.lib.xla_bridge.get_backend().platform}'
    )

    # Add a batch axis to the JAX version.
    self.jax_batch_slice = jax.jit(
        jax.vmap(jax_ops.bilateral_slice, in_axes=0, out_axes=0))

  def test_jax_benchmark(self):
    grid = jax.device_put(self.grid)
    guide = jax.device_put(self.guide)

    f = lambda: self.jax_batch_slice(grid, guide).block_until_ready()
    mean_elapsed_ms, elapsed_ms = _timeit(f, self.burn_iterations,
                                          self.benchmark_iterations)
    print(
        f'JAX batched bilateral_slice took {mean_elapsed_ms} ms per iteration, {elapsed_ms} ms total'
    )


class HdrnetOpsTf2Benchmark(HdrnetOpsBenchmark):

  def setUp(self):
    super().setUp()
    tf.debugging.set_log_device_placement(True)
    print(f'TF2 executing eagerly: {tf.executing_eagerly()}')

  def test_tf2_benchmark(self):
    with tf.device('/GPU:0'):
      grid = tf.convert_to_tensor(self.grid)
      guide = tf.convert_to_tensor(self.guide)

    f = lambda: tf2_ops.bilateral_slice(grid, guide).numpy()
    mean_elapsed_ms, elapsed_ms = _timeit(f, self.burn_iterations,
                                          self.benchmark_iterations)
    print(
        f'TF2 batched bilateral_slice took {mean_elapsed_ms} ms per iteration, {elapsed_ms} ms total'
    )


if __name__ == '__main__':
  tf.test.main()
