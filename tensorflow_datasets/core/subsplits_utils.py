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

"""Even split utils."""

import dataclasses
import enum
import functools
import operator
from typing import List

from absl import logging
from tensorflow_datasets.core import splits as splits_lib
from tensorflow_datasets.core import tfrecords_reader


class Remainder(enum.Enum):
  """Split remainder strategy.

  Attributes:
    LEGACY_PERCENT: Default legacy split strategy, kept for backward
      compatibility. Each splits might not contain the exact same number of
      examples.
    DROP: Examples not divisible by the number of subsplit will be dropped.
      Every subsplit will receives the same number of examples.
    BALANCE: Examples not divisible by the number of subsplit will be
      distributed evenly across subsplits.
    ON_FIRST: Examples not divisible by the number of subsplit will be assigned
      to subsplit 0.
  """
  LEGACY_PERCENT = enum.auto()
  DROP = enum.auto()
  BALANCE = enum.auto()
  ON_FIRST = enum.auto()


@dataclasses.dataclass(frozen=True)
class _EvenSplit(splits_lib.LazySplit):
  """Split matching a subsplit of the given split."""
  split: str
  index: int
  count: int
  remainder: Remainder

  def resolve(self, split_infos: splits_lib.SplitDict) -> splits_lib.SplitArg:
    # Extract the absolute instructions
    # One absolute instruction is created per `+`, so `train[:54%]+train[60%:]`
    # will create 2 absolute instructions.
    read_instruction = tfrecords_reader.ReadInstruction.from_spec(self.split)
    absolute_instructions = read_instruction.to_absolute(split_infos)

    # Create the subsplit
    read_instructions_for_index = [
        self._absolute_to_read_instruction_for_index(abs_inst, split_infos)
        for abs_inst in absolute_instructions
    ]
    return functools.reduce(operator.add, read_instructions_for_index)

  def _absolute_to_read_instruction_for_index(
      self,
      abs_inst,
      split_infos: splits_lib.SplitDict,
  ) -> tfrecords_reader.ReadInstruction:
    start = abs_inst.from_ or 0
    if abs_inst.to is None:  # Note: `abs_inst.to == 0` is valid
      end = split_infos[abs_inst.splitname].num_examples
    else:
      end = abs_inst.to

    assert end >= start, f'start={start}, end={end}'
    num_examples = end - start

    examples_per_host = num_examples // self.count
    shard_start = start + examples_per_host * self.index
    shard_end = start + examples_per_host * (self.index + 1)

    # Handle remaining examples.
    num_unused_examples = num_examples % self.count
    assert num_unused_examples >= 0, num_unused_examples
    assert num_unused_examples < self.count, num_unused_examples
    if num_unused_examples > 0:
      if self.remainder == Remainder.DROP:
        logging.warning('Dropping %d examples of %d examples (host count: %d).',
                        num_unused_examples, num_examples, self.count)
      elif self.remainder == Remainder.BALANCE:
        shard_start += min(self.index, num_unused_examples)
        shard_end += min(self.index + 1, num_unused_examples)
      elif self.remainder == Remainder.ON_FIRST:
        shard_end += num_unused_examples
        if self.index > 0:
          shard_start += num_unused_examples
      else:
        raise ValueError(f'Invalid remainder: {self.remainder}')

    return tfrecords_reader.ReadInstruction(
        abs_inst.splitname,
        from_=shard_start,
        to=shard_end,
        unit='abs',
    )

  def __add__(self, other):
    raise NotImplementedError(
        'Cannot add LazySplit yet. Please open a github issue if you need '
        'this feature.')


def _even_split_percent(split: str, n: int) -> List[str]:
  """Legacy implementation of even_split.

  Example:

  ```python
  assert tfds.even_splits('train', n=3) == [
      'train[0%:33%]', 'train[33%:67%]', 'train[67%:100%]',
  ]
  ```

  Args:
    split: Split name (e.g. 'train', 'test',...)
    n: Number of sub-splits to create

  Returns:
    The list of subsplits.
  """
  if '[' in split or '+' in split:
    raise ValueError('Using tfds.even_splits with subsplits require setting '
                     '`remainder=tfds.Remainder.XYZ`')
  if n <= 0 or n > 100:
    raise ValueError(
        f'n should be > 0 and <= 100. Got {n}. Please use '
        '`remainder=tfds.Remainder.BALANCE` if you need more precise splits.')
  partitions = [round(i * 100 / n) for i in range(n + 1)]
  return [f'{split}[{partitions[i]}%:{partitions[i+1]}%]' for i in range(n)]


def even_splits(
    split: str,
    n: int,
    *,
    remainder: Remainder = Remainder.LEGACY_PERCENT,
) -> List[splits_lib.SplitArg]:
  """Generates a list of non-overlapping sub-splits of same size.

  Example:

  ```python
  split0, split1, split2 = tfds.even_splits(
      'train', n=3, remainder=tfds.Remainder.DROP)

  # Load 1/3 of the train split.
  ds = tfds.load('my_dataset', split=split0)
  ```

  Args:
    split: Split name (e.g. 'train', 'test[75%:]',...)
    n: Number of sub-splits to create
    remainder: Strategy to distribute the remaining examples (if the number of
      examples in the datasets is not evenly divisible by `n`). See
      `tfds.Remainder` for available options.

  Returns:
    The list of subsplits.
  """
  if remainder == Remainder.LEGACY_PERCENT:
    return _even_split_percent(split, n=n)
  return [
      _EvenSplit(split=split, index=i, count=n, remainder=remainder)
      for i in range(n)
  ]
