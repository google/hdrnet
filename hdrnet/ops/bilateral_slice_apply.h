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

#ifndef HDRNET_OPS_BILATERAL_SLICE_APPLY_H_
#define HDRNET_OPS_BILATERAL_SLICE_APPLY_H_

#include "third_party/array/array.h"

namespace hdrnet {

// A fused op that performs:
// - BilateralSlice(grid, guide) -> sliced, a (C, W, H, B) buffer.
// - Reshape(sliced) -> reshaped, a (N, M, W, H, B) buffer of (M x N) matrices.
//   - M * N = C.
// - Multiply(reshaped, input) --> out, a (M, W, H, B) buffer.
//   - input has shape (N, W, H, B) or (N-1, W, H, B).
//   - This is a per-pixel multiply. In the former, it is a linear transform,
//     otherwise, it is affine.
void BilateralSliceApply(nda::array_ref_of_rank<const float, 6> grid,
                         nda::array_ref_of_rank<const float, 3> guide,
                         nda::array_ref_of_rank<const float, 4> input,
                         nda::array_ref_of_rank<float, 4> out);

// Let f(i) be BilateralSliceApply(grid, guide, input), and u(i, j) be the
// codomain tangent vector. We drop the implicit indices (gz, gx, gy, b) for f
// and (x, y, b) for u, guide, and input.
//
// We want to compute vjp(i, j), which is J_f^T * u.
// In this case, J_f is with respect to the two-channel `grid`, so J_f is just
// a "scalar" (it only depends on x, y, and input(x, y, j).
//
// I.e., how much does channel i of the output change with a small change in
// channel (i, j) of `grid`?
// - This depends on:
//   - What the current `input` is.
//   - What the current `guide` is.
// - But is surprisingly, independent of the current value of `grid`!
//   - But this actually makes sense because the output is *linear* in `grid`.
//   - And hence, it only depends on the weights with which you sample `grid`.
void BilateralSliceApplyGridGrad(
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> input,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 6> vjp_out);

// Let f(i) be BilateralSliceApply(grid, guide, input), and u(i) be the
// codomain tangent vector. We drop the implicit indices (gz, gx, gy, b) for f
// and (x, y, b) for u, guide, and input.
//
// We want to compute the scalar vjp_value, which is J_f^T * u.
// In this case, J_f is with respect to the scalar `guide`, so, J_f^T * u =
//   \sum_i[ J_f(i) * u(i) ].
//
// J_f(i) = \partial f(i) / \partial(guide)
//
// I.e., how much does channel i of the output change with a small change in
// `guide`? This depends on:
// - The current value of `input`: it is linear in the input.
// - The current value of `grid`: it is linear in the grid.
// - The current value of `guide`: it is *nonlinear* in the guide: it is used to
//   look up the grid value.
void BilateralSliceApplyGuideGrad(
    nda::array_ref_of_rank<const float, 6> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> input,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 3> vjp_out);

// Let f(i) be BilateralSliceApply(grid, guide, input), and u(i) be the
// codomain tangent vector. We drop the implicit indices (gz, gx, gy, b) for f
// and (x, y, b) for u, guide, and input.
//
// We want to compute vjp(j), which is J_f^T * u =
//   \sum_i[ J_f(i,j) * u(i) ].
//
// In this case, J_f is with respect to `input`:
//   J_f(i,j) = \partial f[i] / \partial(input[j]).
// I.e., how much does channel i of the output change with a small change in
// channel j of the input. This depends on:
// - The current value of `grid`: it is linear in the grid.
// - The current value of `guide`: it is *nonlinear* in the guide: it is used to
//   look up the grid value.
// - It is *independent* of the current value of `input`: that is because f is
//   *linear* in the input.
void BilateralSliceApplyInputGrad(
    nda::array_ref_of_rank<const float, 6> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 4> vjp_out);

}  // namespace hdrnet

#endif  // HDRNET_OPS_BILATERAL_SLICE_APPLY_H_
