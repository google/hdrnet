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

"""Image helpers and operators."""

import numpy as np
import skimage
import skimage.io
import skimage.transform

M_RGB2YUV = np.array([
    [0.2126390, 0.7151688, 0.0721923],
    [0.2126390-1.0, 0.7151688, 0.0721923],
    [0.2126390, 0.7151688, 0.0721923-1.0]])

M_YUV2RGB = np.linalg.inv(M_RGB2YUV)
M_RGB2XYZ = np.array([[0.412453, 0.357580, 0.180423],
                      [0.212671, 0.715160, 0.072169],
                      [0.019334, 0.119193, 0.950227]])
M_XYZ2RGB = np.linalg.inv(M_RGB2XYZ)


# ----- Range transformations -------------------------------------------------
def clamp(image, mini=0.0, maxi=1.0):
  image[image < mini] = mini
  image[image > maxi] = maxi
  return image


def normalize(im):
  mini = np.amin(im)
  maxi = np.amax(im)
  rng = maxi-mini
  im -= mini
  if rng > 0:
    im /= rng
  return im


# ----- Type transformations --------------------------------------------------
def uint8_to_float(image):
  return image.astype(np.float32)/255.0


def float_to_uint8(image):
  image = (clamp(image)*255).astype(np.uint8)
  return image


def uint16_to_float(image):
  return image.astype(np.float32)/32767.0


def int16_to_float(image):
  return image.astype(np.float32)/65535.0


def float_to_int16(image):
  return (image*65535.0).astype(np.int16)


def float_to_uint16(image):
  return (image*32767.0).astype(np.uint16)


# ----- Color transformations -------------------------------------------------
def yuv_to_gray(input_image):
  im = input_image[:, :, 0]  # luminance
  return im


def rgb_to_gray(input_image):
  dtype = input_image.dtype
  im = 0.25*input_image[:, :, 0]
  im += 0.5*input_image[:, :, 1]
  im += 0.25*input_image[:, :, 2]
  im = im.astype(dtype)
  return im


def yuv2rgb(im):
  """Convert image from yuv to rgb.

  Args:
    im: yuv image.
  Returns:
    out: rgb image.
  """

  mtx = M_YUV2RGB

  out = np.zeros(im.shape)
  for i in range(3):
    for j in range(3):
      out[:, :, i] += im[:, :, j]*mtx[i, j]

  return out


# ----- Resampling ------------------------------------------------------------
def rescale(input_image, scale_factor):
  sz = [s*scale_factor for s in input_image.shape[:2]]
  rescaled = skimage.transform.resize(input_image, sz, mode='reflect')
  return rescaled


def resize(input_image, size):
  dtype = input_image.dtype
  ret = skimage.transform.resize(input_image, size)
  if dtype == np.uint8:
    ret = (255*ret).astype(dtype)
  elif dtype == np.uint16:
    ret = (65535*ret).astype(dtype)
  elif dtype == np.float32 or dtype == np.float64:
    ret = ret.astype(dtype)
  else:
    raise ValueError('resize not implemented for type {}'.format(dtype))
  return ret


# ----- I/O -------------------------------------------------------------------
def imread(path):
  return skimage.io.imread(path)


def imwrite(im, path):
  skimage.io.imsave(path, im)
