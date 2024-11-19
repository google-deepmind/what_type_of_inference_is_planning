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

"""Agent forward BP."""
import jax
from jax import numpy as jnp
from jax import random
import numpy as np

from what_type_of_inference_is_planning.agents import agent_base
from what_type_of_inference_is_planning.agents.algorithms import fwd_bp
from what_type_of_inference_is_planning.envs import environment


class AgentFWDBP(agent_base.Agent):
  """Agent forward belief propagation.

  Attributes:
    env: The environment.
    horizon: The horizon length.
    n_actions: Number of actions.
  """

  def __init__(self, env: environment.Environment, horizon: int):
    self.env = env
    self.horizon = horizon
    self.n_actions = env.transition_prob[0].shape[-2]

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
    reward_list = []
    for a in range(self.n_actions):
      reward_list.append(
          fwd_bp.fwdbp(
              x,
              self.env.transition_prob,
              self.env.state_dependencies,
              self.env.reward,
              self.env.reward_dependencies,
              min(self.horizon, remaining_steps + 1),
              actions=a,
              reward_allt=True,
          )[0],
      )
    reward_list = np.array(reward_list)
    p = reward_list == reward_list.max()
    p = p / p.sum()
    _, rng_input = random.split(rng)
    action = random.categorical(rng_input, jnp.log(p))
    print("Reward UBs:", str(reward_list), ", Action chosen:", action)
    return int(action)
