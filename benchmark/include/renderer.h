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

#ifndef RENDERER_H_STXL6FWB
#define RENDERER_H_STXL6FWB

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <GL/glew.h>
#include <GL/freeglut.h>

class Renderer
{
public:
  Renderer(
      int output_width, int output_height,
      int grid_width, int grid_height, int grid_depth,
      std::string vertex_shader, std::string fragment_shader,
      std::string checkpoint_path);
  virtual void upload_input(const cv::Mat &input);
  virtual void render(const float* const coeffs_data, cv::Mat & output,
      double *upload_coeff_time, double *draw_time, double *readback_time);
  virtual ~Renderer ();

protected:
  virtual void load_guide_parameters(std::string checkpoint_path) = 0;
  virtual void gl_extra_setup() = 0;
  virtual void upload_coefficients(const float* const coeffs_data) = 0;

  int output_width_;
  int output_height_;

  int grid_width_;
  int grid_height_;
  int grid_depth_;

  GLuint program_;
  GLuint vertex_shader_;
  GLuint fragment_shader_;

  GLuint input_texture_;

  GLuint output_texture_;
  GLuint framebuffer_;

  GLuint coeffs_textures_[3];

  // Timer counter queries.
  static const int kNumQueries = 4;
  GLuint query_ids_[kNumQueries];

  float pVertex_[8] = { 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f };
  float pTexCoord_[8] = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f };
};


class StandardRenderer : public Renderer
{
public:
  explicit StandardRenderer(
      int output_width, int output_height,
      int grid_width, int grid_height, int grid_depth,
      std::string vertex_shader, std::string fragment_shader,
      std::string checkpoint_path);


protected:
  virtual void load_guide_parameters(std::string checkpoint_path) override;
  virtual void gl_extra_setup() override;
  virtual void upload_coefficients(const float* const coeffs_data) override;

};


class MultiscaleRenderer : public Renderer
{
  // TODO: we should render the lower resolutions to textures and upsample on GPU,
  // instead of sampling at the finest resolution: in gpyrnn.frag
public:
  explicit MultiscaleRenderer(
      int output_width, int output_height,
      int grid_width, int grid_height, int grid_depth,
      std::string vertex_shader, std::string fragment_shader,
      std::string checkpoint_path);
  virtual void upload_input(const cv::Mat &input) override;

protected:
  virtual void load_guide_parameters(std::string checkpoint_path) override;
  virtual void gl_extra_setup() override;
  virtual void upload_coefficients(const float* const coeffs_data) override;

  GLuint coeffs_textures_[9];
  GLuint input_texture_ds2_;
  GLuint input_texture_ds4_;
};

#endif /* end of include guard: RENDERER_H_STXL6FWB */

