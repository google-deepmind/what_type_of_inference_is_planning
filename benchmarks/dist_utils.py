# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utility functions for generating reward distributions."""

from collections.abc import Sequence
from typing import Any, Dict, List


def get_reward(
    case: Dict[str, Any], variables: Sequence[str], state: str
) -> float:
  """Read the unnormalized reward from factored tree.

  Args:
    case: A tree that stores the reward of the corresponding state values.
    variables: The list of variables this reward depend on.
    state: A string indicating the state of each variable in variables.

  Returns:
    The unnormalized reward.
  """
  state_dict = {}
  for idx, svar in enumerate(variables):
    if (state[idx] == '1'):
      state_dict[svar] = 'true'
    else:
      state_dict[svar] = 'false'
  parent = case
  finished_search = False
  updated = False
  maxiter = len(state)
  count = 0
  rwd = 0.
  while (not finished_search and count <= maxiter):
    for key_p in parent.keys():
      for key_c in parent[key_p].keys():
        if (key_c == state_dict[key_p]):
          parent = parent[key_p][key_c]
          count += 1
          updated = True
          if not isinstance(parent, dict):
            rwd = float(parent)
            finished_search = True
          break
      if updated:
        updated = False
      else:
        finished_search = True

  return rwd


def generate_reward_distribution(
    var_num: int,
    case: Dict[str, Any],
    variables: Sequence[str],
    state: str = '',
) -> List[float]:
  """Recursively generate reward table on states.

  Args:
    var_num: Number of variables.
    case: A tree that stores the reward of the corresponding state values.
    variables: The list of variables this reward depend on.
    state: A string indicating the state of each variable in variables.

  Returns:
    The reward table on states.
  """
  if (var_num == 0):
    rwd = get_reward(case, variables, state)
    return rwd
  else:
    return [
        generate_reward_distribution(var_num - 1, case, variables, state + '0'),
        generate_reward_distribution(var_num - 1, case, variables, state + '1')
    ]


