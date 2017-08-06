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

#version 430
// Standard shader for curve-based guidance map
precision lowp float;

layout(location = 1) uniform sampler2D sRGB;
layout(location = 2) uniform sampler3D sAffineGridRow0;
layout(location = 3) uniform sampler3D sAffineGridRow1;
layout(location = 4) uniform sampler3D sAffineGridRow2;

uniform mat3x4 uGuideCcm;
uniform vec3 uGuideShifts[16];
uniform vec3 uGuideSlopes[16];
uniform vec4 uMixMatrix;

in vec2 texCoord;
layout(location = 0) out vec3 colorOut;

void main() {
  vec2 offset_x = vec2(1.0, 0.0)/2048.0;
  vec2 offset_y = vec2(0.0, 1.0)/2048.0;
  vec4 rgba = vec4(texture(sRGB, texCoord).xyz, 1.0);

  // Compute guide.
  vec3 tmp = (rgba*uGuideCcm);
  vec3 tmp2 = vec3(0);
  for (int i = 0; i < 16; ++i) {
    tmp2 += uGuideSlopes[i].rgb * max(vec3(0), tmp - uGuideShifts[i].rgb);
  }
  float guide = clamp(dot(vec4(tmp2, 1.0), uMixMatrix), 0, 1);
  
  // Sample coefficients
  vec3 gridLoc = vec3(texCoord.x, texCoord.y, guide);
  vec4 row0 = texture(sAffineGridRow0, gridLoc);
  vec4 row1 = texture(sAffineGridRow1, gridLoc);
  vec4 row2 = texture(sAffineGridRow2, gridLoc);

  // Render
  colorOut = clamp(vec3(dot(row0, rgba), dot(row1, rgba), dot(row2, rgba)), 0.0, 1.0);
}
