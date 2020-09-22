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

#include "bilateral_slice_apply.h"

#include <algorithm>
#include <cmath>

#include "numerics.h"

namespace hdrnet {

void BilateralSliceApply(nda::array_ref_of_rank<const float, 6> grid,
                         nda::array_ref_of_rank<const float, 3> guide,
                         nda::array_ref_of_rank<const float, 4> input,
                         nda::array_ref_of_rank<float, 4> out) {
  // - Samples centered at 0.5.
  // - Repeating boundary conditions.
  const int grid_input_channels = grid.shape().dim(0).extent();
  const int grid_depth = grid.shape().dim(2).extent();
  const int grid_width = grid.shape().dim(3).extent();
  const int grid_height = grid.shape().dim(4).extent();
  const int input_channels = input.shape().dim(0).extent();
  const int input_width = input.shape().dim(1).extent();
  const int input_height = input.shape().dim(2).extent();
  const float scale_x = static_cast<float>(grid_width) / input_width;
  const float scale_y = static_cast<float>(grid_height) / input_height;

  nda::for_all_indices(out.shape(), [&](int i, int x, int y, int b) {
    const float gxf = (x + 0.5f) * scale_x;
    const float gyf = (y + 0.5f) * scale_y;
    // TODO(jiawen): Offset gz by 0.5 as well.
    const float gzf = guide(x, y, b) * grid_depth;

    const int gx0 = static_cast<int>(std::floor(gxf - 0.5f));
    const int gy0 = static_cast<int>(std::floor(gyf - 0.5f));
    const int gz0 = static_cast<int>(std::floor(gzf - 0.5f));

    float value = 0.0f;
    for (int j = 0; j < grid_input_channels; ++j) {
      // Grid trilinear interpolation to retrieve grid(gxf, gyf, gzf, i, j).
      float grid_sample = 0.0f;
      for (int gy = gy0; gy < gy0 + 2; ++gy) {
        const int gyc = std::clamp(gy, 0, grid_height - 1);
        const float wy = LerpWeight(gy + 0.5f, gyf);

        for (int gx = gx0; gx < gx0 + 2; ++gx) {
          const int gxc = std::clamp(gx, 0, grid_width - 1);
          const float wx = LerpWeight(gx + 0.5f, gxf);

          for (int gz = gz0; gz < gz0 + 2; ++gz) {
            const int gzc = std::clamp(gz, 0, grid_depth - 1);
            const float wz = SmoothedLerpWeight(gz + 0.5f, gzf);

            grid_sample += wx * wy * wz * grid(j, i, gzc, gxc, gyc, b);
          }  // gz
        }    // gy
      }      // gx
      // Grid trilinear interpolation.

      // Matrix multiply.
      if (j < input_channels) {
        value += grid_sample * input(j, x, y, b);
      } else {  // Offset term
        value += grid_sample;
      }
    }
    out(i, x, y, b) = value;
  });
}

void BilateralSliceApplyGridGrad(
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> input,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 6> vjp_out) {
  const int grid_depth = vjp_out.shape().dim(2).extent();
  const int grid_width = vjp_out.shape().dim(3).extent();
  const int grid_height = vjp_out.shape().dim(4).extent();
  const int input_channels = input.shape().dim(0).extent();
  const int input_width = input.shape().dim(1).extent();
  const int input_height = input.shape().dim(2).extent();
  const float scale_x = static_cast<float>(input_width) / grid_width;
  const float scale_y = static_cast<float>(input_height) / grid_height;

  nda::for_all_indices(vjp_out.shape(), [&](int j, int i, int gz, int gx,
                                            int gy, int b) {
    const int x0 = static_cast<int>(std::floor(scale_x * (gx + 0.5f - 1.0f)));
    const int x1_exclusive =
        static_cast<int>(std::ceil(scale_x * (gx + 0.5f + 1.0f)));
    const int y0 = static_cast<int>(std::floor(scale_y * (gy + 0.5f - 1.0f)));
    const int y1_exclusive =
        static_cast<int>(std::ceil(scale_y * (gy + 0.5f + 1.0f)));

    float vjp_value = 0.0f;
    for (int y = y0; y < y1_exclusive; ++y) {
      const int y_mirror = MirrorBoundary(y, input_height);
      const float gyf = (y + 0.5f) / scale_y;
      const float wy = LerpWeight(gy + 0.5f, gyf);

      for (int x = x0; x < x1_exclusive; ++x) {
        // TODO(jiawen): Consider using clamp boundary.
        const int x_mirror = MirrorBoundary(x, input_width);
        const float gxf = (x + 0.5f) / scale_x;
        const float wx = LerpWeight(gx + 0.5f, gxf);

        // TODO(jiawen): Offset gz by 0.5 as well.
        const float gzf = guide(x_mirror, y_mirror, b) * grid_depth;
        float wz = SmoothedLerpWeight(gz + 0.5f, gzf);
        if ((gz == 0 && gzf < 0.5f) ||
            (gz == grid_depth - 1 && gzf > grid_depth - 0.5f)) {
          wz = 1.0f;
        }

        // Index input accounting for optional offset.
        const float input_value =
            (j < input_channels) ? input(j, x_mirror, y_mirror, b) : 1.0f;
        const float grad_value = wx * wy * wz * input_value;

        vjp_value += grad_value * codomain_tangent(i, x_mirror, y_mirror, b);
      }  // y
    }    // x
    vjp_out(j, i, gz, gx, gy, b) = vjp_value;
  });
}

void BilateralSliceApplyGuideGrad(
    nda::array_ref_of_rank<const float, 6> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> input,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 3> vjp_out) {
  const int grid_input_channels = grid.shape().dim(0).extent();
  const int output_channels = grid.shape().dim(1).extent();
  const int grid_depth = grid.shape().dim(2).extent();
  const int grid_width = grid.shape().dim(3).extent();
  const int grid_height = grid.shape().dim(4).extent();
  const int input_channels = input.shape().dim(0).extent();
  const int input_width = input.shape().dim(1).extent();
  const int input_height = input.shape().dim(2).extent();
  const float scale_x = static_cast<float>(grid_width) / input_width;
  const float scale_y = static_cast<float>(grid_height) / input_height;

  nda::for_all_indices(vjp_out.shape(), [&](int x, int y, int b) {
    const float gxf = (x + 0.5f) * scale_x;
    const float gyf = (y + 0.5f) * scale_y;
    // TODO(jiawen): Offset gz by 0.5 as well.
    const float gzf = guide(x, y, b) * grid_depth;

    const int gx0 = static_cast<int>(std::floor(gxf - 0.5f));
    const int gy0 = static_cast<int>(std::floor(gyf - 0.5f));
    const int gz0 = static_cast<int>(std::floor(gzf - 0.5f));

    float vjp_value = 0.0f;
    for (int i = 0; i < output_channels; ++i) {
      float grad_value = 0.0f;

      for (int j = 0; j < grid_input_channels; ++j) {
        float grid_sample = 0.0f;

        // Grid trilinear interpolation to retrieve grid(gxf, gyf, gzf, i, j).
        for (int gy = gy0; gy < gy0 + 2; ++gy) {
          const int gyc = std::clamp(gy, 0, grid_height - 1);
          const float wy = LerpWeight(gy + 0.5f, gyf);

          for (int gx = gx0; gx < gx0 + 2; ++gx) {
            const int gxc = std::clamp(gx, 0, grid_width - 1);
            const float wx = LerpWeight(gx + 0.5f, gxf);

            for (int gz = gz0; gz < gz0 + 2; ++gz) {
              const int gzc = std::clamp(gz, 0, grid_depth - 1);
              // TODO(jiawen): Offset gz by 0.5 as well?
              const float dwz =
                  grid_depth * SmoothedLerpWeightGrad(gz + 0.5f, gzf);

              grid_sample += wx * wy * dwz * grid(j, i, gzc, gxc, gyc, b);
            }  // gz
          }    // gy
        }      // gx
        // Grid trilinear interpolation.

        const float input_value =
            (j < input_channels) ? input(j, x, y, b) : 1.0f;
        grad_value += grid_sample * input_value;
      }  // Sum over j.

      vjp_value += grad_value * codomain_tangent(i, x, y, b);
    }  // Sum over i.
    vjp_out(x, y, b) = vjp_value;
  });
}

void BilateralSliceApplyInputGrad(
    nda::array_ref_of_rank<const float, 6> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 4> vjp_out) {
  const int output_channels = grid.shape().dim(1).extent();
  const int grid_depth = grid.shape().dim(2).extent();
  const int grid_width = grid.shape().dim(3).extent();
  const int grid_height = grid.shape().dim(4).extent();
  const int guide_width = guide.shape().dim(0).extent();
  const int guide_height = guide.shape().dim(1).extent();
  const float scale_x = static_cast<float>(grid_width) / guide_width;
  const float scale_y = static_cast<float>(grid_height) / guide_height;

  nda::for_all_indices(vjp_out.shape(), [&](int j, int x, int y, int b) {
    const float gxf = (x + 0.5f) * scale_x;
    const float gyf = (y + 0.5f) * scale_y;
    // TODO(jiawen): Offset gz by 0.5 as well.
    const float gzf = guide(x, y, b) * grid_depth;

    const int gx0 = static_cast<int>(std::floor(gxf - 0.5f));
    const int gy0 = static_cast<int>(std::floor(gyf - 0.5f));
    const int gz0 = static_cast<int>(std::floor(gzf - 0.5f));

    float vjp_value = 0.0f;
    for (int i = 0; i < output_channels; ++i) {
      float grad_val = 0.0f;
      // Grid trilinear interpolation to retrieve grid(gxf, gyf, gzf, i, j).
      for (int gy = gy0; gy < gy0 + 2; ++gy) {
        const int gyc = std::clamp(gy, 0, grid_height - 1);
        const float wy = LerpWeight(gy + 0.5f, gyf);

        for (int gx = gx0; gx < gx0 + 2; ++gx) {
          const int gxc = std::clamp(gx, 0, grid_width - 1);
          const float wx = LerpWeight(gx + 0.5f, gxf);

          for (int gz = gz0; gz < gz0 + 2; ++gz) {
            const int gzc = std::clamp(gz, 0, grid_depth - 1);
            const float wz = SmoothedLerpWeight(gz + 0.5f, gzf);

            grad_val += wx * wy * wz * grid(j, i, gzc, gxc, gyc, b);
          }  // gz
        }    // gy
      }      // gx
      // Grid trilinear interpolation.

      vjp_value += grad_val * codomain_tangent(i, x, y, b);
    }  // sum over i.
    vjp_out(j, x, y, b) = vjp_value;
  });
}

}  // namespace hdrnet
