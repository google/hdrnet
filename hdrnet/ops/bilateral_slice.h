// Copyright 2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HDRNET_OPS_BILATERAL_SLICE_H_
#define HDRNET_OPS_BILATERAL_SLICE_H_

#include "third_party/array/array.h"

namespace hdrnet {

// `BilateralSlice` implements the slice operation from HDRnet (go/hdrnet). It
// takes as input:
// - A bilateral grid `grid`, an array with shape
//   (grid_channels, grid_depth, grid_width, grid_height, batch_size)
// - A guide map, an array with shape
//   (width, height, batch_size)

// The bilateral grid and guide map share an underlying x and y domain, but are
// sampled differently - the grid is typically much coarser. The values of the
// guide map are indices in the grid's z axis.

// "Slicing" a bilateral grid using a guide map produces an array with shape
// (width, height, batch_size, grid_channels). Each grid in a batch is sliced
// independently. For each guide map, at each (x, y) location, we compute its
// corresponding grid coordinates:
//   gx = (x + 0.5) * grid_width / width
//   gy = (y + 0.5) * grid_height / height
//   gz = guide[x, y] * grid_depth
// We sample grid[:, gz, gx, gy, gz, b] using trilinear interpolation.
void BilateralSlice(nda::array_ref_of_rank<const float, 5> grid,
                    nda::array_ref_of_rank<const float, 3> guide,
                    nda::array_ref_of_rank<float, 4> out);

// Let f(c) be BilateralSlice(grid, guide), and u(c) be the
// codomain tangent vector. We drop the implicit indices (gz, gx, gy, b) for f
// and (x, y, b) for u and guide.
//
// We want to compute vjp(c), which is J_f^T * u.
// In this case, J_f is with respect to the scalar `grid` (because we are
// slicing out one channel at a time), so J_f is a "scalar" (it only depends on
// x, y, and guide(x, y)).
//
// I.e., how much does channel c of the output change with a small change in
// channel c of `grid`?
// - This depends *only* on the current value of `guide`.
// - It is surprisingly, independent of the current value of `grid`!
//   - But this actually makes sense because the output is *linear* in `grid`.
//   - And hence, it only depends on the weights with which you sample `grid`.
void BilateralSliceGridGrad(
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 5> grid_vjp_out);

// Let f(c) be BilateralSlice(grid, guide), and u(c) be the
// codomain tangent vector. We drop the implicit indices (gz, gx, gy b) for f
// and (x, y, b) for u and guide.
//
// We want to compute the scalar vjp_value, which is J_f^T * u.
// In this case, J_f is with respect to the scalar `guide`, so, J_f^T * u =
//   \sum_i[ J_f(c) * u(c) ].
//
// J_f(c) = \partial f(c) / \partial(guide)
//
// I.e., how much does channel c of the output change with a small change in
// `guide`? This depends on:
// - The current value of `grid`: it is linear in the grid.
// - The current value of `guide`: it is *nonlinear* in the guide: it is used to
//   look up the grid value.
void BilateralSliceGuideGrad(
    nda::array_ref_of_rank<const float, 5> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 3> guide_vjp_out);

}  // namespace hdrnet

#endif  // HDRNET_OPS_BILATERAL_SLICE_H_
