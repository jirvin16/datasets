# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors.
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

"""penguins dataset."""

import csv

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

# TODO(b/171746535): Add penguins_raw dataset
_PENGUINS_CSV = ('https://storage.googleapis.com/download.tensorflow.org/data/'
                 'palmer_penguins/penguins_size.csv')
_DESCRIPTION = """
Measurements for three penguin species observed in the Palmer Archipelago, Antarctica.

These data were collected from 2007 - 2009 by Dr. Kristen Gorman with the [Palmer
Station Long Term Ecological Research Program](https://pal.lternet.edu/), part
of the [US Long Term Ecological Research Network](https://lternet.edu/). The
data were originally imported from the [Environmental Data
Initiative](https://environmentaldatainitiative.org/) (EDI) Data Portal, and are
available for use by CC0 license ("No Rights Reserved") in accordance with the
Palmer Station Data Policy. This copy was imported from [Allison Horst's GitHub
repository](https://allisonhorst.github.io/palmerpenguins/articles/intro.html).

The curated dataset contains 7 variables (n = 344 penguins):

 * species
 * island
 * culmen_length_mm
 * culmen_depth_mm
 * flipper_length_mm
 * body_mass_g
 * sex
"""

_CITATION = """
@Manual{,
  title = {palmerpenguins: Palmer Archipelago (Antarctica) penguin data},
  author = {Allison Marie Horst and Alison Presmanes Hill and Kristen B Gorman},
  year = {2020},
  note = {R package version 0.1.0},
  doi = {10.5281/zenodo.3960218},
  url = {https://allisonhorst.github.io/palmerpenguins/},
}
"""


class Penguins(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for penguins dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'species':
                tfds.features.ClassLabel(
                    names=['Adelie', 'Chinstrap', 'Gentoo']),
            'island':
                tfds.features.ClassLabel(names=['Biscoe', 'Dream', 'Torgersen']
                                        ),
            'culmen_length_mm':
                tf.float32,
            'culmen_depth_mm':
                tf.float32,
            'flipper_length_mm':
                tf.float32,
            'body_mass_g':
                tf.float32,
            'sex':
                tfds.features.ClassLabel(names=['FEMALE', 'MALE', 'NA']),
        }),
        supervised_keys=None,
        homepage='https://allisonhorst.github.io/palmerpenguins/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.download(_PENGUINS_CSV)

    return {
        'train': self._generate_examples(path),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    with path.open() as f:
      for i, row in enumerate(csv.DictReader(f)):
        for k in row:
          # 'sex' includes some malformed values.
          if k == 'sex':
            if row[k] not in ('MALE', 'FEMALE'):
              row[k] = 'NA'
          else:
            row[k] = row[k].replace('NA', 'NaN')
        yield i, row
