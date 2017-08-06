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

"""Loads and writes dataset metadata to a .json record."""

import json
import os


def write_dataset_meta(path, nsamples, fname_to_timestamp_map):
  nsamples_path = os.path.join(path, 'nsamples.json')
  meta = {'nsamples': nsamples}
  f = open(nsamples_path, 'w')
  json.dump(meta, f, indent=2)
  f.close()

  timestamps_path = os.path.join(path, 'timestamps.json')
  f = open(timestamps_path, 'w')
  json.dump(fname_to_timestamp_map, f, indent=2, sort_keys=True)
  f.close()


def get_dataset_meta(path):
  nsamples_path = os.path.join(path, 'nsamples.json')
  timestamps_path = os.path.join(path, 'timestamps.json')
  f = open(nsamples_path, 'r')
  meta = json.load(f)
  f.close()

  f = open(timestamps_path, 'r')
  fname_to_timestamp_map = json.load(f)
  f.close()

  return (meta, fname_to_timestamp_map)
