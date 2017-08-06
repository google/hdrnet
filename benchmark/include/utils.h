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

#ifndef UTILS_H_JWUMPG6C
#define UTILS_H_JWUMPG6C

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>

cv::Mat load_image(std::string input_path);

void shader_from_file(const std::string filename, GLuint& shader);

void load_binary_data(std::string filename, int length, float* output);

#endif /* end of include guard: UTILS_H_JWUMPG6C */
