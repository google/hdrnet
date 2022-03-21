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

#include "bilateral_slice.h"

#include <algorithm>
#include <cmath>

#include "numerics.h"
#include "third_party/array/array.h"

namespace hdrnet {

void BilateralSlice(nda::array_ref_of_rank<const float, 5> grid,
                    nda::array_ref_of_rank<const float, 3> guide,
                    nda::array_ref_of_rank<float, 4> out) {
  // - Samples centered at 0.5f.
  // - Repeating boundary conditions.
  const int grid_depth = grid.dim<1>().extent();
  const int grid_width = grid.dim<2>().extent();
  const int grid_height = grid.dim<3>().extent();
  const float scale_x = static_cast<float>(grid_width) / guide.width();
  const float scale_y = static_cast<float>(grid_height) / guide.height();

  nda::for_all_indices(out.shape(), [&](int c, int x, int y, int b) {
    const float gxf = (x + 0.5f) * scale_x;
    const float gyf = (y + 0.5f) * scale_y;
    // Because 0.5f applied afterwards in calculating gz0 and wz, the effective
    // depth index is:
    //    guide * grid_depth + 0.5f
    const float gzf = guide(x, y, b) * grid_depth;

    const int gx0 = static_cast<int>(std::floor(gxf - 0.5f));
    const int gy0 = static_cast<int>(std::floor(gyf - 0.5f));
    const int gz0 = static_cast<int>(std::floor(gzf - 0.5f));

    // Grid trilinear interpolation to retrieve grid(gxf, gyf, gzf, i, j).
    float value = 0.0f;
    for (int gy = gy0; gy < gy0 + 2; ++gy) {
      const int gyc = std::clamp(gy, 0, grid_height - 1);
      const float wy = LerpWeight(gy + 0.5f, gyf);

      for (int gx = gx0; gx < gx0 + 2; ++gx) {
        const int gxc = std::clamp(gx, 0, grid_width - 1);
        const float wx = LerpWeight(gx + 0.5f, gxf);

        for (int gz = gz0; gz < gz0 + 2; ++gz) {
          const int gzc = std::clamp(gz, 0, grid_depth - 1);
          const float wz = SmoothedLerpWeight(gz + 0.5f, gzf);

          value += wx * wy * wz * grid(c, gzc, gxc, gyc, b);
        }
      }
    }
    // Grid trilinear interpolation.

    out(c, x, y, b) = value;
  });
}

void BilateralSliceGridGrad(
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 5> grid_vjp_out) {
  const int grid_depth = grid_vjp_out.dim<1>().extent();
  const int grid_width = grid_vjp_out.dim<2>().extent();
  const int grid_height = grid_vjp_out.dim<3>().extent();
  const int guide_width = guide.width();
  const int guide_height = guide.height();
  const float scale_x = static_cast<float>(guide.width()) / grid_width;
  const float scale_y = static_cast<float>(guide.height()) / grid_height;

  nda::for_all_indices(grid_vjp_out.shape(), [&](int gc, int gz, int gx, int gy,
                                                 int b) {
    const int x0 = static_cast<int>(std::floor(scale_x * (gx + 0.5f - 1.0f)));
    const int x1_exclusive =
        static_cast<int>(std::ceil(scale_x * (gx + 0.5f + 1.0f)));
    const int y0 = static_cast<int>(std::floor(scale_y * (gy + 0.5f - 1.0f)));
    const int y1_exclusive =
        static_cast<int>(std::ceil(scale_y * (gy + 0.5f + 1.0f)));

    float vjp_value = 0.0f;
    for (int y = y0; y < y1_exclusive; ++y) {
      const int y_mirror = MirrorBoundary(y, guide_height);
      const float gyf = (y + 0.5f) / scale_y;
      const float wy = LerpWeight(gy + 0.5f, gyf);

      for (int x = x0; x < x1_exclusive; ++x) {
        // TODO(jiawen): Consider using clamp boundary.
        const int x_mirror = MirrorBoundary(x, guide_width);
        const float gxf = (x + 0.5f) / scale_x;
        const float wx = LerpWeight(gx + 0.5f, gxf);

        const float gzf = guide(x_mirror, y_mirror, b) * grid_depth;
        float wz = SmoothedLerpWeight(gz + 0.5f, gzf);
        if ((gz == 0 && gzf < 0.5f) ||
            (gz == grid_depth - 1 && gzf > grid_depth - 0.5f)) {
          wz = 1.0f;
        }

        vjp_value += wz * wx * wy * codomain_tangent(gc, x_mirror, y_mirror, b);
      }  // y
    }    // x

    grid_vjp_out(gc, gz, gx, gy, b) = vjp_value;
  });
}

void BilateralSliceGuideGrad(
    nda::array_ref_of_rank<const float, 5> grid,
    nda::array_ref_of_rank<const float, 3> guide,
    nda::array_ref_of_rank<const float, 4> codomain_tangent,
    nda::array_ref_of_rank<float, 3> guide_vjp_out) {
  const int grid_channels = grid.dim<0>().extent();
  const int grid_depth = grid.dim<1>().extent();
  const int grid_width = grid.dim<2>().extent();
  const int grid_height = grid.dim<3>().extent();
  const float scale_x = static_cast<float>(grid_width) / guide.width();
  const float scale_y = static_cast<float>(grid_height) / guide.height();

  nda::for_all_indices(guide_vjp_out.shape(), [&](int x, int y, int b) {
    const float gxf = (x + 0.5f) * scale_x;
    const float gyf = (y + 0.5f) * scale_y;
    const float gzf = guide(x, y, b) * grid_depth;

    const int gx0 = static_cast<int>(std::floor(gxf - 0.5f));
    const int gy0 = static_cast<int>(std::floor(gyf - 0.5f));
    const int gz0 = static_cast<int>(std::floor(gzf - 0.5f));

    float vjp_value = 0.0f;
    for (int c = 0; c < grid_channels; ++c) {
      float grid_sample = 0.0f;

      // Grid trilinear interpolation to retrieve grid(c, gzf, gxf, gyf, gzf).
      for (int gy = gy0; gy < gy0 + 2; ++gy) {
        const int gyc = std::clamp(gy, 0, grid_height - 1);
        const float wy = LerpWeight(gy + 0.5f, gyf);

        for (int gx = gx0; gx < gx0 + 2; ++gx) {
          const int gxc = std::clamp(gx, 0, grid_width - 1);
          const float wx = LerpWeight(gx + 0.5f, gxf);

          for (int gz = gz0; gz < gz0 + 2; ++gz) {
            const int gzc = std::clamp(gz, 0, grid_depth - 1);
            const float dwz =
                grid_depth * SmoothedLerpWeightGrad(gz + 0.5f, gzf);

            grid_sample += wx * wy * dwz * grid(c, gzc, gxc, gyc, b);
          }
        }
      }
      vjp_value += grid_sample * codomain_tangent(c, x, y, b);
    }  // Sum over c.

    guide_vjp_out(x, y, b) = vjp_value;
  });
}

}  // namespace hdrnet
