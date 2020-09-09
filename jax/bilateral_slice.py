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
"""JAX implementation of bilateral_slice and its vjp."""

import jax
import jax.numpy as jnp
from .numerics import lerp_weight
from .numerics import smoothed_lerp_weight
from .numerics import smoothed_lerp_weight_grad


# TODO(jiawen): Cache intermediates and pass them in.
def bilateral_slice_guide_vjp(grid, guide, codomain_tangent):
  """VJP for bilateral_slice with respect to `guide`.

  Note that this depends on both `grid` and `guide`.

  Args:
    grid: The bilateral grid with shape (gh, gw, gd, gc).
    guide: The guide image with shape (h, w).
    codomain_tangent: The codomain tangent with shape (gh, gw, gc).

  Returns:
    The vector-Jacobian product codomain_tangent * J^T(grid, guide) with shape
    (h, w) (the same as that of the primal `guide`).

  """
  ii, jj = jnp.meshgrid(
      jnp.arange(guide.shape[0]), jnp.arange(guide.shape[1]), indexing='ij')

  scale_i = grid.shape[0] / guide.shape[0]
  scale_j = grid.shape[1] / guide.shape[1]
  grid_depth = grid.shape[2]

  gif = (ii + 0.5) * scale_i
  gjf = (jj + 0.5) * scale_j
  gkf = guide * grid.shape[2]

  # Compute trilinear interpolation weights without clamping.
  gi0 = jnp.floor(gif - 0.5).astype(jnp.int32)
  gj0 = jnp.floor(gjf - 0.5).astype(jnp.int32)
  gk0 = jnp.floor(gkf - 0.5).astype(jnp.int32)
  gi1 = gi0 + 1
  gj1 = gj0 + 1
  gk1 = gk0 + 1

  wi0 = lerp_weight(gi0, gif)
  wi1 = lerp_weight(gi1, gif)
  wj0 = lerp_weight(gj0, gjf)
  wj1 = lerp_weight(gj1, gjf)
  dwk0 = grid_depth * smoothed_lerp_weight_grad(gk0, gkf)
  dwk1 = grid_depth * smoothed_lerp_weight_grad(gk1, gkf)

  w_000 = wi0 * wj0 * dwk0
  w_001 = wi0 * wj0 * dwk1
  w_010 = wi0 * wj1 * dwk0
  w_011 = wi0 * wj1 * dwk1
  w_100 = wi1 * wj0 * dwk0
  w_101 = wi1 * wj0 * dwk1
  w_110 = wi1 * wj1 * dwk0
  w_111 = wi1 * wj1 * dwk1

  # But clip when indexing into `grid`.
  gi0c = gi0.clip(0, grid.shape[0] - 1)
  gj0c = gj0.clip(0, grid.shape[1] - 1)
  gk0c = gk0.clip(0, grid.shape[2] - 1)

  gi1c = (gi0 + 1).clip(0, grid.shape[0] - 1)
  gj1c = (gj0 + 1).clip(0, grid.shape[1] - 1)
  gk1c = (gk0 + 1).clip(0, grid.shape[2] - 1)

  #        ijk: 0 means floor, 1 means ceil.
  grid_val_000 = grid[gi0c, gj0c, gk0c, :]
  grid_val_001 = grid[gi0c, gj0c, gk1c, :]
  grid_val_010 = grid[gi0c, gj1c, gk0c, :]
  grid_val_011 = grid[gi0c, gj1c, gk1c, :]
  grid_val_100 = grid[gi1c, gj0c, gk0c, :]
  grid_val_101 = grid[gi1c, gj0c, gk1c, :]
  grid_val_110 = grid[gi1c, gj1c, gk0c, :]
  grid_val_111 = grid[gi1c, gj1c, gk1c, :]

  # Append a singleton "channels" dimension.
  w_000, w_001, w_010, w_011, w_100, w_101, w_110, w_111 = jnp.atleast_3d(
      w_000, w_001, w_010, w_011, w_100, w_101, w_110, w_111)

  # Do multiply-add with trilerp weights.
  # grid_val has shape (guide_height, guide_width, channels).
  grid_val = (
      jnp.multiply(w_000, grid_val_000) + jnp.multiply(w_001, grid_val_001) +
      jnp.multiply(w_010, grid_val_010) + jnp.multiply(w_011, grid_val_011) +
      jnp.multiply(w_100, grid_val_100) + jnp.multiply(w_101, grid_val_101) +
      jnp.multiply(w_110, grid_val_110) + jnp.multiply(w_111, grid_val_111))

  # Multiply element-wise then sum over the channels axis.
  return jnp.sum(jnp.multiply(grid_val, codomain_tangent), axis=-1)


def _compute_scale_pad(image_extent, grid_extent):
  """Computes spatial scale and padding given image and grid extents.

  Args:
    image_extent: Image extent along that axis.
    grid_extent: Grid extent along the corresponding axis.

  Returns:
    (scale, half_pad)
    scale: spatial scaling (image pixels per grid cell)
    half_pad: how much padding is needed on either side of the image.
  """
  scale = image_extent / grid_extent
  half_scale = 0.5 * scale
  half_pad = jnp.ceil(half_scale).astype(jnp.int32)

  return scale, half_pad


def _compute_spatial_weights(image_extent, grid_extent):
  """Computes spatial weights given image and grid extents.

  Args:
    image_extent: Image extent along that axis.
    grid_extent: Grid extent along the corresponding axis.

  Returns:
    An (image_extent, grid_extent) array with the spatial weight for each
    spatial and grid position.
  """
  scale, half_pad = _compute_scale_pad(image_extent, grid_extent)
  image_extent_padded = image_extent + 2 * half_pad

  indices = jnp.arange(image_extent_padded) - half_pad
  grid_indices_float = (indices + 0.5) / scale

  gif, gi = jnp.meshgrid(
      grid_indices_float, jnp.arange(grid_extent), indexing='ij')

  weights = lerp_weight(gi, gif)

  return weights


def _symmetric_pad_ij(image, grid_shape):
  """Symmetrically pads an image along the first two axes.

  Args:
    image: the image.
    grid_shape: shape of the corresponding bilateral grid.

  Returns:
    `image` padded along the first two axes in "symmetric" mode sufficient
    to compute the proper grid gradient for the given image and grid shape.
  """
  _, half_pad_i = _compute_scale_pad(image.shape[0], grid_shape[0])
  _, half_pad_j = _compute_scale_pad(image.shape[1], grid_shape[1])

  # Pad along i and j only. Set the rest to 0.
  pads = [(half_pad_i, half_pad_i), (half_pad_j, half_pad_j)] + [(0, 0)] * (
      len(image.shape) - 2)

  return jnp.pad(image, pads, mode='symmetric')


def _compute_range_weights(guide, grid_shape):
  """Computes range weights for the given guide image and grid shape.

  Args:
    guide: The guide image with shape (h, w).
    grid_shape: The grid shape, an array-like containing [gh, gw, gd, gc].

  Returns:
    An (image_extent, grid_extent) array with the spatial weight for each
    spatial and grid position.
  """
  guide_padded = _symmetric_pad_ij(guide, grid_shape)

  # Rescale `image` from [0, 1] to [0, grid_depth].
  # These are the floating point k coordinates of each sample.
  grid_depth = grid_shape[2]
  gk_float = guide_padded * grid_depth

  # Each sample with float value kf can splat onto locations:
  # k0 = floor(kf - 0.5)
  # k1 = ceil(kf - 0.5)
  #
  # The subtraction by 0.5 is necessary:
  # - Grid samples are located at half-integer coordinates:
  #   k = 0 places its sample at kf = 0.5.
  # - If kf = 1.4, the tent weight function is nonzero in the range [0.4, 1.4].
  #   Therefore, we need to splat to k0 = 0 and k1 = 1.
  # - If kf = 1.9, the tent weight function is nonzero in the range [0.9, 1.9].
  #   Therefore, we need to splat to k0 = 1 and k1 = 2.
  gk_floor = jnp.floor(gk_float - 0.5)
  gk_ceil = jnp.ceil(gk_float - 0.5)

  # Compute tent weights before clipping.
  wk_floor = smoothed_lerp_weight(gk_floor, gk_float)
  wk_ceil = smoothed_lerp_weight(gk_ceil, gk_float)

  # Cast to int for indexing.
  gk_floor = gk_floor.astype(jnp.int32)
  gk_ceil = gk_ceil.astype(jnp.int32)

  # Handle boundary conditions:
  # - Set the weight to 0 where the tent weight is positive but outside
  #   [0, grid_depth].
  # - Set the weight to 1 where the sample is between [0, 0.5) and
  #   (depth - 0.5, depth].
  wk_floor = jnp.where((gk_ceil == 0) & (gk_float < 0.5), 0, wk_floor)
  wk_ceil = jnp.where(
      (gk_floor == grid_depth - 1) & (gk_float > grid_depth - 0.5), 0, wk_ceil)
  wk_ceil = jnp.where((gk_ceil == 0) & (gk_float < 0.5), 1, wk_ceil)
  wk_floor = jnp.where(
      (gk_floor == grid_depth - 1) & (gk_float > grid_depth - 0.5), 1, wk_floor)

  # Now clip int coordinates for splatting. Coordinates outside [0, grid_depth)
  # will have zero weight so splatting to them does nothing.
  gk_floor_clipped = gk_floor.clip(0, grid_depth - 1)
  gk_ceil_clipped = gk_ceil.clip(0, grid_depth - 1)

  # Compute the i and j indices where we want to splat the weights wk with +=.
  # grid[ii, jj, gk_floor] += wk_floor
  # grid[ii, jj, gk_ceil] += wk_ceil
  ii, jj = jnp.meshgrid(
      jnp.arange(guide_padded.shape[0]),
      jnp.arange(guide_padded.shape[1]),
      indexing='ij')

  range_weights = jnp.zeros(
      (guide_padded.shape[0], guide_padded.shape[1], grid_depth))
  range_weights = jax.ops.index_add(range_weights,
                                    jax.ops.index[ii, jj,
                                                  gk_floor_clipped], wk_floor)
  range_weights = jax.ops.index_add(range_weights,
                                    jax.ops.index[ii, jj,
                                                  gk_ceil_clipped], wk_ceil)

  return range_weights


def bilateral_slice_grid_vjp(guide, codomain_tangent, grid_shape):
  """VJP for bilateral_slice with respect to `grid`.

  Note that this is independent of `grid`.

  Args:
    guide: The guide image with shape (h, w).
    codomain_tangent: The codomain tangent with shape (gh, gw, gc).
    grid_shape: The grid shape, an array-like containing [gh, gw, gd, gc].

  Returns:
    The vector-Jacobian product codomain_tangent * J^T(guide) with shape
    (gh, gw, gd, gc) (the same as that of the primal `grid`).
  """
  w_i = _compute_spatial_weights(guide.shape[0], grid_shape[0])
  w_j = _compute_spatial_weights(guide.shape[1], grid_shape[1])
  w_k = _compute_range_weights(guide, grid_shape)

  codomain_tangent = _symmetric_pad_ij(codomain_tangent, grid_shape)

  # After padding, guide and codomain_tangent have spatial extents (h', w').
  #
  # `w_i` has shape (h', gh) and indexed with [i, a].
  # w_i[i, a] is the spatial weight on the first axis for pixel position i on
  # grid position a.
  #
  # `w_j` has shape (w', gw) and indexed with [j, b].
  # w_j[j, b] is the spatial weight on the second axis for pixel position j on
  # grid position b.
  #
  # `w_k` has shape (h', w', gd) and indexed with [i, j, c].
  # w_k[i, j, c] is the range weight (the third axis) for pixel position (i, j)
  # on grid position c.
  #
  # `codomain_tangent` has shape (h', w', gc) and indexed with [i, j, d].
  #
  # Use einsum to compute the weighted summation over the spatial axes,
  # returning a grid.
  return jnp.einsum('ia,jb,ijc,ijd->abcd', w_i, w_j, w_k, codomain_tangent)


# TODO(jiawen): Migrate bilateral_slice_np to use this math.
@jax.custom_vjp
def bilateral_slice(grid, guide):
  """Slices a bilateral grid using the a guide image.

  Args:
    grid: The bilateral grid with shape (gh, gw, gd, gc).
    guide: A guide image with shape (h, w). Values must be in the range [0, 1].

  Returns:
    sliced: An image with shape (h, w, gc), computed by trilinearly
    interpolating for each grid channel c the grid at 3D position
    [(i + 0.5) * gh / h,
     (j + 0.5) * gw / w,
     guide(i, j) * gd]
  """
  ii, jj = jnp.meshgrid(
      jnp.arange(guide.shape[0]), jnp.arange(guide.shape[1]), indexing='ij')

  scale_i = grid.shape[0] / guide.shape[0]
  scale_j = grid.shape[1] / guide.shape[1]

  gif = (ii + 0.5) * scale_i
  gjf = (jj + 0.5) * scale_j
  gkf = guide * grid.shape[2]

  # Compute trilinear interpolation weights without clamping.
  gi0 = jnp.floor(gif - 0.5).astype(jnp.int32)
  gj0 = jnp.floor(gjf - 0.5).astype(jnp.int32)
  gk0 = jnp.floor(gkf - 0.5).astype(jnp.int32)
  gi1 = gi0 + 1
  gj1 = gj0 + 1
  gk1 = gk0 + 1

  wi0 = lerp_weight(gi0, gif)
  wi1 = lerp_weight(gi1, gif)
  wj0 = lerp_weight(gj0, gjf)
  wj1 = lerp_weight(gj1, gjf)
  wk0 = smoothed_lerp_weight(gk0, gkf)
  wk1 = smoothed_lerp_weight(gk1, gkf)

  w_000 = wi0 * wj0 * wk0
  w_001 = wi0 * wj0 * wk1
  w_010 = wi0 * wj1 * wk0
  w_011 = wi0 * wj1 * wk1
  w_100 = wi1 * wj0 * wk0
  w_101 = wi1 * wj0 * wk1
  w_110 = wi1 * wj1 * wk0
  w_111 = wi1 * wj1 * wk1

  # But clip when indexing into `grid`.
  gi0c = gi0.clip(0, grid.shape[0] - 1)
  gj0c = gj0.clip(0, grid.shape[1] - 1)
  gk0c = gk0.clip(0, grid.shape[2] - 1)

  gi1c = (gi0 + 1).clip(0, grid.shape[0] - 1)
  gj1c = (gj0 + 1).clip(0, grid.shape[1] - 1)
  gk1c = (gk0 + 1).clip(0, grid.shape[2] - 1)

  #        ijk: 0 means floor, 1 means ceil.
  grid_val_000 = grid[gi0c, gj0c, gk0c, :]
  grid_val_001 = grid[gi0c, gj0c, gk1c, :]
  grid_val_010 = grid[gi0c, gj1c, gk0c, :]
  grid_val_011 = grid[gi0c, gj1c, gk1c, :]
  grid_val_100 = grid[gi1c, gj0c, gk0c, :]
  grid_val_101 = grid[gi1c, gj0c, gk1c, :]
  grid_val_110 = grid[gi1c, gj1c, gk0c, :]
  grid_val_111 = grid[gi1c, gj1c, gk1c, :]

  # Append a singleton "channels" dimension.
  w_000, w_001, w_010, w_011, w_100, w_101, w_110, w_111 = jnp.atleast_3d(
      w_000, w_001, w_010, w_011, w_100, w_101, w_110, w_111)

  # TODO(jiawen): Cache intermediates and pass them in.
  # Just pass out w_ijk and the same ones multiplied by by dwk.
  return (jnp.multiply(w_000, grid_val_000) +
          jnp.multiply(w_001, grid_val_001) +
          jnp.multiply(w_010, grid_val_010) +
          jnp.multiply(w_011, grid_val_011) +
          jnp.multiply(w_100, grid_val_100) +
          jnp.multiply(w_101, grid_val_101) +
          jnp.multiply(w_110, grid_val_110) +
          jnp.multiply(w_111, grid_val_111))


def _bilateral_slice_fwd(grid, guide):
  return bilateral_slice(grid, guide), (grid, guide, grid.shape)


def _bilateral_slice_bwd(res, codomain_tangent):
  grid, guide, grid_shape = res
  grid_vjp = bilateral_slice_grid_vjp(guide, codomain_tangent, grid_shape)
  guide_vjp = bilateral_slice_guide_vjp(grid, guide, codomain_tangent)
  return grid_vjp, guide_vjp


# Register custom VJP for backprop.
bilateral_slice.defvjp(_bilateral_slice_fwd, _bilateral_slice_bwd)
