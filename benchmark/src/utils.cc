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

#include "utils.h"

cv::Mat load_image(std::string input_path) {
  cv::Mat image = cv::imread(input_path, CV_LOAD_IMAGE_COLOR);
  cv::Mat image_rgb;
  cv::cvtColor(image, image_rgb, CV_BGR2RGB, 3);
  return image_rgb;
}

void load_binary_data(std::string filename, int length, float* output) {
  std::ifstream file;
  file.open(filename, std::ios::in);
  if(!file) {
    std::cout << "Failed to load file " << filename << std::endl;
    throw;
  }
  file.read((char*) output, sizeof(float)*length);
  file.close();
}

void shader_from_file(const std::string filename, GLuint& shader) {
  std::ifstream file;
  file.open(filename, std::ios::in);
  if(!file) {
    std::cout << "Failed to load file " << filename << std::endl;
    throw;
  }
  std::string content;
  file.seekg(0, std::ios::end);
  content.reserve(file.tellg());
  file.seekg(0, std::ios::beg);
  content.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());

  const GLchar* source = (const GLchar*) content.c_str();
  glShaderSource(shader, 1, &source, NULL);
  glCompileShader(shader);

  GLint shader_success;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &shader_success);
  if (shader_success == GL_FALSE) {
    std::cout << "Failed to compile shader" << std::endl;
    GLint logSize = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logSize);
    std::vector<GLchar> errorLog(logSize);
    glGetShaderInfoLog(shader, logSize, &logSize, &errorLog[0]);
    std::cout << errorLog.data() << std::endl;
    glDeleteShader(shader);
    throw;
  }
}
