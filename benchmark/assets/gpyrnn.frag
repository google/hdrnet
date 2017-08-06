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
precision lowp float;

layout(location = 1) uniform sampler2D sRGB;
layout(location = 2) uniform sampler2D sRGBds2;
layout(location = 3) uniform sampler2D sRGBds4;

layout(location = 4) uniform sampler3D sAffineGridRow0;
layout(location = 5) uniform sampler3D sAffineGridRow1;
layout(location = 6) uniform sampler3D sAffineGridRow2;
layout(location = 7) uniform sampler3D sAffineGridRow3;
layout(location = 8) uniform sampler3D sAffineGridRow4;
layout(location = 9) uniform sampler3D sAffineGridRow5;
layout(location = 10) uniform sampler3D sAffineGridRow6;
layout(location = 11) uniform sampler3D sAffineGridRow7;
layout(location = 12) uniform sampler3D sAffineGridRow8;

uniform vec4 uGuideConv1[16*3];
uniform float uGuideConv2[17*3];

in vec2 texCoord;
layout(location = 0) out vec3 colorOut;

float sigmoid(float x) {
  return 1.0f/(1.0f+exp(-x));
}

void main() {
  vec4 rgba[3]; 
  rgba[0] = vec4(texture(sRGB, texCoord).xyz, 1.0);
  rgba[1] = vec4(texture(sRGBds2, texCoord/2).xyz, 1.0); // 2x2 downsampled
  rgba[2] = vec4(texture(sRGBds4, texCoord/4).xyz, 1.0); // 4x4 downsampled

  float guide_conv1[3*16];

  for(int c = 0; c < 3; ++c) {
    for(int i = 0; i < 16; ++i) {
      guide_conv1[i + 16*c] = max(dot(rgba[c], uGuideConv1[i + 16*c]), 0.0);
    }
  }

  float guide_conv2[3] = {0, 0, 0};
  for(int c = 0; c < 3; ++c) {
    for(int i = 0; i < 16; ++i) {
      guide_conv2[c] += guide_conv1[i + 16*c]*uGuideConv2[i + 17*c];
    }
    guide_conv2[c] += uGuideConv2[16];
    guide_conv2[c] = sigmoid(guide_conv2[c]);
  }

  // Sample coefficients (coefficients are in reverse orders w.r.t. to guide and image)
  vec3 gridLoc = vec3(texCoord.x, texCoord.y, guide_conv2[2]);
  vec4 row0 = texture(sAffineGridRow0, gridLoc);
  vec4 row1 = texture(sAffineGridRow1, gridLoc);
  vec4 row2 = texture(sAffineGridRow2, gridLoc);
  vec3 level2 = vec3(dot(row0, rgba[2]), dot(row1, rgba[2]), dot(row2, rgba[2]));

  gridLoc = vec3(texCoord.x, texCoord.y, guide_conv2[1]);
  row0 = texture(sAffineGridRow3, gridLoc);
  row1 = texture(sAffineGridRow4, gridLoc);
  row2 = texture(sAffineGridRow5, gridLoc);
  vec3 level1 = vec3(dot(row0, rgba[1]), dot(row1, rgba[1]), dot(row2, rgba[1]));

  gridLoc = vec3(texCoord.x, texCoord.y, guide_conv2[0]);
  row0 = texture(sAffineGridRow6, gridLoc);
  row1 = texture(sAffineGridRow7, gridLoc);
  row2 = texture(sAffineGridRow8, gridLoc);
  vec3 level0 = vec3(dot(row0, rgba[0]), dot(row1, rgba[0]), dot(row2, rgba[0]));

  // Render
  colorOut = clamp(level0+level1+level2, 0, 1);
  // colorOut = vec3(rgba[2].rgb);
}
