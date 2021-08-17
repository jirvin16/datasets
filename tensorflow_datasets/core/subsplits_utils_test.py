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

"""Tests for subsplits_utils."""

import pytest
from tensorflow_datasets.core import splits as splits_lib
from tensorflow_datasets.core import subsplits_utils

DROP = subsplits_utils.Remainder.DROP
BALANCE = subsplits_utils.Remainder.BALANCE
ON_FIRST = subsplits_utils.Remainder.ON_FIRST


def test_even_splits_legacy():
  assert subsplits_utils.even_splits(
      'train', n=3) == ['train[0%:33%]', 'train[33%:67%]', 'train[67%:100%]']

  assert subsplits_utils.even_splits('train', 4) == [
      'train[0%:25%]', 'train[25%:50%]', 'train[50%:75%]', 'train[75%:100%]'
  ]

  with pytest.raises(ValueError):
    subsplits_utils.even_splits('train', 0)
  with pytest.raises(ValueError):
    subsplits_utils.even_splits('train', 101)


@pytest.mark.parametrize(
    'num_examples, n, remainder, expected',
    [
        (9, 1, DROP, ['']),  # Full split selected
        (9, 2, DROP, ['[:4]', '[4:8]']),
        (9, 3, DROP, ['[:3]', '[3:6]', '[6:9]']),
        (9, 4, DROP, ['[:2]', '[2:4]', '[4:6]', '[6:8]']),  # Last ex dropped
        (11, 2, DROP, ['[:5]', '[5:10]']),
        (11, 3, DROP, ['[:3]', '[3:6]', '[6:9]']),  # Last 2 exs dropped
        (9, 1, BALANCE, ['']),
        (9, 2, BALANCE, ['[:5]', '[5:9]']),  # split0 has extra ex
        (9, 3, BALANCE, ['[:3]', '[3:6]', '[6:9]']),
        (9, 4, BALANCE, ['[:3]', '[3:5]', '[5:7]', '[7:9]']),  # 0 has extra ex
        (11, 3, BALANCE, ['[:4]', '[4:8]', '[8:11]']),  # 0, 1 have extra ex
        (11, 4, BALANCE, ['[:3]', '[3:6]', '[6:9]', '[9:11]']),
        (9, 1, ON_FIRST, ['']),
        (9, 2, ON_FIRST, ['[:5]', '[5:9]']),
        (9, 3, ON_FIRST, ['[:3]', '[3:6]', '[6:9]']),
        (9, 4, ON_FIRST, ['[:3]', '[3:5]', '[5:7]', '[7:9]']),
        (11, 3, ON_FIRST, ['[:5]', '[5:8]', '[8:11]']),  # 0 has 2 extra exs
        (11, 4, ON_FIRST, ['[:5]', '[5:7]', '[7:9]', '[9:11]']),
    ],
)
def test_even_splits(num_examples, n, remainder, expected):
  split_infos = splits_lib.SplitDict(
      [
          splits_lib.SplitInfo(
              name='train',
              shard_lengths=[num_examples],
              num_bytes=0,
          ),
      ],
      dataset_name='mnist',
  )

  subsplits = subsplits_utils.even_splits('train', n, remainder=remainder)

  file_instructions = [split_infos[s].file_instructions for s in subsplits]
  expected_file_instructions = [
      split_infos[f'train{s}'].file_instructions for s in expected
  ]
  assert file_instructions == expected_file_instructions


def test_even_splits_subsplit():
  split_infos = splits_lib.SplitDict(
      [
          splits_lib.SplitInfo(
              name='train',
              shard_lengths=[2, 3, 2, 3],  # 10
              num_bytes=0,
          ),
          splits_lib.SplitInfo(
              name='test',
              shard_lengths=[8],
              num_bytes=0,
          ),
      ],
      dataset_name='mnist',
  )

  # Test to split multiple splits
  subsplits = subsplits_utils.even_splits(
      'train+test[50%:]', 3, remainder=ON_FIRST)

  expected = [
      'train[:4]+test[4:6]',
      'train[4:7]+test[6:7]',
      'train[7:]+test[7:8]',
  ]

  file_instructions = [split_infos[s].file_instructions for s in subsplits]
  expected_file_instructions = [
      split_infos[s].file_instructions for s in expected
  ]
  assert file_instructions == expected_file_instructions
