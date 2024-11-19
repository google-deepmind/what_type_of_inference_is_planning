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

"""Convert spudd files to npz files."""
# pylint: disable=g-explicit-length-test  (for better readability)
import os
import sys
from typing import Any, Dict, Sequence, Tuple, List
from unittest import mock

sys.path.append("../third_party/AISPFS/src")
sys.modules["factor_graph"] = mock.Mock()  # module not needed in third party.
sys.modules["pgmax"] = mock.Mock()  # module not needed in third party.

import numpy as np  # pylint: disable=g-import-not-at-top
import spudd_parser
import utils
from benchmarks import dist_utils


def get_enums_info(file_name: str) -> Tuple[List[str], List[str]]:
  """Get enum information for specific tasks.

  Args:
    file_name: Name of the spudd file.

  Returns:
    The states that should sum to one.
    The states that should sum to less or equal to one.
  """
  enums_info = []
  if file_name.startswith("elevators"):
    enums_info = ["elevator_at_floor__e0", "elevator_at_floor__e1"]

  enums_less1_info = []
  if file_name.startswith("crossing_traffic"):
    enums_less1_info = ["robot_at"]

  return enums_info, enums_less1_info


def save_factor_info(
    init_state: Dict[str, Sequence[int]],
    state_dependency: Dict[str, Sequence[str]],
    valid_actions: Sequence[str],
    atomic_action_lst: Sequence[str],
    reward_dist: Dict[str, Any],
    trans_prob: Dict[str, Any],
    s_vars: Sequence[str],
    a_vars: Sequence[str],
    inst_name: str,
):
  """Save factor information to npz file.

  Args:
    init_state: The initial state of the environment.
    state_dependency: What each state depends on in previous time step.
    valid_actions: The valid action combinations.
    atomic_action_lst: Name of the valid actions.
    reward_dist: A tree of reward values for each action state combination.
    trans_prob: The transition probability of each state from previous step.
    s_vars: Name of state variables.
    a_vars: Name of aciton variables.
    inst_name: Instance name.
  """
  default_act = "noop"
  len_of_cases = len(reward_dist[default_act]["parents"])

  spudd_dict = {}
  s_name_to_idx = {}
  init_state_array = np.zeros(len(s_vars), dtype=int)
  for cs_idx, s_var in enumerate(s_vars):
    s_name_to_idx[s_var] = cs_idx

  spudd_dict["state_names"] = s_vars
  spudd_dict["state_names_id"] = s_name_to_idx

  for state in init_state:
    value = init_state[state][1]
    if value == 1:
      init_state_array[s_name_to_idx[state]] = 1
  spudd_dict["init_state"] = init_state_array
  s_dependency_to_idx = []
  for _, s_var in enumerate(s_vars):
    dependency_idx = []
    for s_var_parent in state_dependency[s_var + "'"]:
      dependency_idx.append(s_name_to_idx[s_var_parent])
    s_dependency_to_idx.append(dependency_idx)

  spudd_dict["state_dependencies"] = s_dependency_to_idx

  r_dependency_to_idx = []
  for case in reward_dist[default_act]["parents"]:
    dependency_idx = []
    for r_var_parent in case:
      dependency_idx.append(s_name_to_idx[r_var_parent])
    r_dependency_to_idx.append(dependency_idx)

  spudd_dict["reward_dependencies"] = r_dependency_to_idx

  spudd_dict["action_names"] = atomic_action_lst

  time_step = 1
  time_stamp = str(time_step)
  next_time = time_step + 1
  next_stamp = str(next_time)
  prev_time = time_step - 1
  prev_stamp = str(prev_time)

  trans_factor_vars = []

  for _, s_var in enumerate(s_vars):
    child_var = s_var + "'"
    factor_var = []

    for curr_s_var in state_dependency[child_var]:
      factor_var.append("t" + prev_stamp + "_" + curr_s_var)

    factor_var.append("t" + prev_stamp + "_atomic_action")
    factor_var.append("t" + time_stamp + "_" + s_var)

    trans_factor_vars.append(factor_var)
  trans_matrices = []

  for landing_s, factor_var in zip(s_vars, trans_factor_vars):
    parent_s = state_dependency[landing_s + "'"]
    trans_f_dist = utils.generate_factor_dist(
        time=time_step,
        state_vars=parent_s,
        action_vars=a_vars,
        valid_actions=valid_actions,
        atomic_action_lst=atomic_action_lst,
        factor_type="trans",
        variables=factor_var,
        dist=trans_prob,
        mes_type="fw",
        a_mask="",
    )
    trans_matrices.append(trans_f_dist.tolist())

  spudd_dict["transition_prob"] = trans_matrices

  partial_rwd_s_factor_vars = []
  for idx in range(0, len_of_cases):
    factor_var = []
    for svar in reward_dist[default_act]["parents"][idx]:
      factor_var.append("t" + time_stamp + "_" + svar)

    factor_var.append("t" + next_stamp + "_pr" + str(idx + 1))
    partial_rwd_s_factor_vars.append(factor_var)

  reward_matrices_origin = []
  for case, variables in zip(
      reward_dist[default_act]["cases"],
      reward_dist[default_act]["parents"],
  ):
    pr_s_f_dist_origin = dist_utils.generate_reward_distribution(
        var_num=len(variables), case=case, variables=variables
    )
    reward_matrices_origin.append(pr_s_f_dist_origin)

  spudd_dict["reward"] = reward_matrices_origin

  n_ent = len(s_vars)
  transition_prob, state_dependencies = np.empty(n_ent, dtype=object), np.empty(
      n_ent, dtype=object
  )
  for i in range(n_ent):
    transition_prob[i] = np.array(spudd_dict["transition_prob"][i])
    state_dependencies[i] = np.array(spudd_dict["state_dependencies"][i])
  n_rew = len(spudd_dict["reward"])
  reward, reward_dependencies = np.empty(n_rew, dtype=object), np.empty(
      n_rew, dtype=object
  )
  for i in range(n_rew):
    reward[i] = np.array(spudd_dict["reward"][i])
    reward_dependencies[i] = np.array(spudd_dict["reward_dependencies"][i])
  reward = np.array(reward, dtype=object)
  reward_dependencies = np.array(reward_dependencies, dtype=object)

  enums_info, enums_less1_info = get_enums_info(inst_name)

  enums = []
  for var_prefix in enums_info:
    enum_group = []
    for idx, var in enumerate(s_vars):
      if var.startswith(var_prefix):
        enum_group.append(idx)
    if len(enum_group) > 0:
      enums.append(np.array(enum_group))

  enums_less1 = []
  for var_prefix in enums_less1_info:
    enum_group = []
    for idx, var in enumerate(s_vars):
      if var.startswith(var_prefix):
        enum_group.append(idx)
    if len(enum_group) > 0:
      enums_less1.append(np.array(enum_group))
  enums = np.array(enums, dtype=object)
  enums_less1 = np.array(enums_less1, dtype=object)

  np.savez_compressed(
      f"npz/{inst_name}.npz",
      transition_prob=transition_prob,
      state_dependencies=state_dependencies,
      reward=reward,
      reward_dependencies=reward_dependencies,
      init_state=init_state_array,
      enums=enums,
      enums_less1=enums_less1,
      state_names=spudd_dict["state_names"],
      state_names_id=spudd_dict["state_names_id"],
      action_names=spudd_dict["action_names"],
  )

  return


def convert_spudd_to_npz(spudd_file_path: str):
  """Convert spudd file to npz file."""
  # Do not soften transition probability.
  def no_soften(_, val):
    return val
  spudd_parser.SPUDD_Parser._soften = no_soften  # pylint: disable=protected-access
  parsed_spudd = spudd_parser.SPUDD_Parser(spudd_file_path)
  size_info = parsed_spudd.get_problem_size()
  state_vars = parsed_spudd.get_state_vars()
  action_vars = parsed_spudd.get_action_vars()

  s_num = len(state_vars)
  a_num = len(action_vars)
  print(
      "Saving npz for spudd {},{} state variable and {} action variable."
      " State dependency maximum {},minimum {}, average {}. Valid action {}."
      " Enumeration maximum {}, minimum {},average {}".format(
          spudd_file_path,
          s_num,
          a_num,
          size_info[0],
          size_info[1],
          size_info[2],
          size_info[3],
          size_info[4],
          size_info[5],
          size_info[6],
      )
  )

  s_vars = parsed_spudd.get_state_vars()
  a_vars = parsed_spudd.get_action_vars()
  valid_actions = parsed_spudd.get_valid_action_lst()
  trans_prob = parsed_spudd.get_pseudo_trans()
  state_dependency = parsed_spudd.get_state_dependency()
  reward_dist = parsed_spudd.get_reward_table()
  atomic_action_lst = parsed_spudd.get_atomic_action_lst()
  init_state = parsed_spudd.get_init_state()
  save_factor_info(
      init_state=init_state,
      state_dependency=state_dependency,
      valid_actions=valid_actions,
      atomic_action_lst=atomic_action_lst,
      reward_dist=reward_dist,
      trans_prob=trans_prob,
      s_vars=s_vars,
      a_vars=a_vars,
      inst_name=parsed_spudd.get_inst_name(),
  )


def main():
  """Convert all spudd files to npz files."""
  spudd_dir = "spudd_sperseus/"

  for spudd_file in os.listdir(spudd_dir):
    if spudd_file.endswith(".spudd"):
      convert_spudd_to_npz(spudd_dir + spudd_file)


if __name__ == "__main__":
  main()
