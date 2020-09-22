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

# Lint as: python3
"""Numerics for bilateral_slice."""

import jax.numpy as jnp


def lerp_weight(x, xs):
  """Linear interpolation weight from a sample at x to xs.

  Returns the linear interpolation weight of a "query point" at coordinate `x`
  with respect to a "sample" at coordinate `xs`.

  The integer coordinates `x` are at pixel centers.
  The floating point coordinates `xs` are at pixel edges.
  (OpenGL convention).

  Args:
    x: "Query" point position.
    xs: "Sample" position.

  Returns:
    - 1 when x = xs.
    - 0 when |x - xs| > 1.
  """
  dx = x - xs
  abs_dx = abs(dx)
  return jnp.maximum(1.0 - abs_dx, 0.0)


def smoothed_abs(x, eps=1e-8):
  """A smoothed version of |x| with improved numerical stability."""
  return jnp.sqrt(jnp.multiply(x, x) + eps)


def smoothed_abs_grad(x, eps=1e-8):
  """Gradient of SmoothedAbs with respect to x.

  This is a smoothed version of sign(x) with improved numerical stability.

  Args:
    x: the argument.
    eps: a small number.

  Returns:
    x / (sqrt(x * x) + eps).
  """
  return jnp.divide(x, jnp.sqrt(jnp.multiply(x, x) + eps))


def smoothed_lerp_weight(x, xs, eps=1e-8):
  """Smoothed version of `LerpWeight` with gradients more suitable for backprop.

  Let f(x, xs) = LerpWeight(x, xs)
               = max(1 - |x - xs|, 0)
               = max(1 - |dx|, 0)

  f is not smooth when:
  - |dx| is close to 0. We smooth this by replacing |dx| with
    SmoothedAbs(dx, eps) = sqrt(dx * dx + eps), which has derivative
    dx / sqrt(dx * dx + eps).
  - |dx| = 1. When smoothed, this happens when dx = sqrt(1 - eps). Like ReLU,
    We just ignore this (in the implementation below, when the floats are
    exactly equal, we choose the SmoothedAbsGrad path since it is more useful
    than returning a 0 gradient).

  Args:
    x: "Query" point position.
    xs: "Sample" position.
    eps: a small number.

  Returns:
    max(1 - |dx|, 0) where |dx| is smoothed_abs(dx).
  """
  dx = x - xs
  abs_dx = smoothed_abs(dx, eps)
  return jnp.maximum(1.0 - abs_dx, 0.0)


def smoothed_lerp_weight_grad(x, xs, eps=1e-8):
  """Gradient of smoothed_lerp_weight."""
  dx = x - xs
  abs_dx = smoothed_abs(dx, eps)
  grad = smoothed_abs_grad(dx, eps)
  return jnp.where(abs_dx > 1, 0, grad)
