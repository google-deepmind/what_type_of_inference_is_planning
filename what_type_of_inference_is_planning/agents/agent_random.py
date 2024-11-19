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

"""Random agent."""
import jax
from jax import random
import numpy as np

from what_type_of_inference_is_planning.agents import agent_base
from what_type_of_inference_is_planning.envs import environment


class AgentRandom(agent_base.Agent):
  """Random agent.

  Attributes:
    n_actions: Number of actions.
  """

  def __init__(self, env: environment.Environment):
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
    _, rng_input = random.split(rng)
    action = random.randint(rng_input, [1], 0, self.n_actions)[0]
    return int(action)
