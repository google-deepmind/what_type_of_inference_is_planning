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

"""VILP Agent."""
from absl import logging
import jax
from jax import numpy as jnp
from jax import random
import numpy as np

from what_type_of_inference_is_planning.agents import agent_base
from what_type_of_inference_is_planning.agents.algorithms import bin_planner
from what_type_of_inference_is_planning.envs import environment


class AgentVILP(agent_base.Agent):
  """Agent variational inference with LP solver.

  Attributes:
    env: The environment.
    horizon: The horizon length.
    n_actions: Number of actions.
    wiring: Pre-multiplying matrices. See bin_planner for detail.
    t_mat: Transition matrices for the dynamics.
    r_mat: Analogous to the transition matrices for the rewards.
  """

  def __init__(self, env: environment.Environment, horizon: int):
    self.env = env
    self.horizon = horizon
    self.n_actions = env.transition_prob[0].shape[-2]
    n_ent = len(env.transition_prob)
    n_rew = len(env.reward)
    n_act = env.transition_prob[0].shape[-2]
    f_t_mat, b_t_mat = bin_planner.get_mat_time(env.state_dependencies)
    f_r_mat, b_r_mat = bin_planner.get_mat_time(env.reward_dependencies)

    self.wiring = (f_t_mat, b_t_mat, f_r_mat, b_r_mat)
    self.t_mat = [
        env.transition_prob[i].reshape(-1, n_act, 2) for i in range(n_ent)
    ]
    self.r_mat = [env.reward[i].reshape(-1) for i in range(n_rew)]

  def observe(
      self, rng: jax.Array, x: np.ndarray, remaining_steps: int
  ) -> int:
    """Observe the environment.

    Args:
      rng: Random number.
      x: Current state.
      remaining_steps: Number of remaining steps.

    Returns:
      action: Action chosen.
    """
    init_t_mat = bin_planner.get_init(self.env.state_dependencies, x)
    init_r_mat = bin_planner.get_init(self.env.reward_dependencies, x)
    init = [init_t_mat, init_r_mat]
    reward_list = []
    for a in range(self.n_actions):
      logging.info("action %d/%d", a, self.n_actions)
      reward_list.append(
          bin_planner.solve_bin(
              self.t_mat,
              self.r_mat,
              self.wiring,
              init,
              min(self.horizon, remaining_steps + 1),
              self.env.enums,
              self.env.enums_less1,
              None,
              a,
          )[0]
      )
    reward_list = np.array(reward_list)

    p = reward_list == reward_list.max()
    p = p / p.sum()
    _, rng_input = random.split(rng)
    action = random.categorical(rng_input, jnp.log(p))

    logging.info(
        "*** Rewards: %s, Action chosen: %d", str(reward_list), action
    )
    return int(action)
