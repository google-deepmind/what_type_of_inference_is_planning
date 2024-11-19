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

"""Environment for evaluation."""
# pylint: disable=g-explicit-length-test  (for better readability)
from typing import List
from absl import logging
import jax
from jax import random
import numpy as np

from what_type_of_inference_is_planning.agents import agent_base


class Environment:
  """Environment for evaluating agents.

  Attributes:
    transition_prob: The transition probability for each state.
    state_dependencies: The dependencies for each state.
    reward: The reward for each sub reward value.
    reward_dependencies: The dependencies for each sub reward.
    init_state: The initial state of the environment.
    enums: The states that should sum to 1.
    enums_less1: The states that should sum to less or equal to 1.
    action_names: The names of the actions.
    state_names: The names of the states.
    x: Current state of the environment.
  """

  def __init__(self, file: str):
    """Initialize and resets the environment.

    Args:
      file: The npz file path that stores the environment information.
    """
    with open(file, "rb") as fin:
      with np.load(fin, allow_pickle=True) as f:
        self.transition_prob = f["transition_prob"]
        self.state_dependencies = f["state_dependencies"]
        self.reward = f["reward"]
        self.reward_dependencies = f["reward_dependencies"]
        self.init_state = f["init_state"]
        self.enums = f["enums"]
        self.enums_less1 = f["enums_less1"]
        self.action_names = f["action_names"]
        self.state_names = f["state_names"]

    self.reset()

  def reset(self):
    """Reset the environment."""
    self.x = self.init_state

  def advance(
      self,
      rng: jax.Array,
      action: int,
      rands_to_use: List[float] | None = None,
  ):
    """Advance the environment.

    Args:
      rng: Random number.
      action: The action the agent chosen.
      rands_to_use: Use list of pre-generated random numbers for the environment
        if not None.
    """
    new_x = []
    for i in range(len(self.transition_prob)):
      if len(self.state_dependencies[i]) > 0:
        prevx = tuple(self.x[self.state_dependencies[i]])
      else:
        prevx = ()
      p = self.transition_prob[i][prevx + (action, 1)].astype(float)
      rng, rng_input = random.split(rng)
      rand = random.uniform(rng_input, shape=p.shape)

      if rands_to_use is not None:
        rand = rands_to_use[i]
      new_x.append(int(rand < p))
    self.x = np.array(new_x)
    return

  def current_reward(self) -> float:
    """Return the current reward."""
    reward = 0.0
    for i in range(len(self.reward)):
      if len(self.reward_dependencies[i]) > 0:
        x = tuple(self.x[self.reward_dependencies[i]])
      else:
        x = ()
      reward += self.reward[i][x]
    return reward

  def run(
      self,
      rng: jax.Array,
      agent: agent_base.Agent,
      n_extra_steps: int,
      rands_list_to_use: List[List[float]] | None = None,
  ) -> float:
    """Run agent in environment.

    Args:
      rng: Random number.
      agent: Agent to run in environment.
      n_extra_steps: Number of extra steps to run.
      rands_list_to_use: Use list of pre-generated random numbers for the
        environment per step if not None.

    Returns:
      reward_sum: The sum of the reward.

    """
    current_reward = self.current_reward()
    reward_sum = current_reward
    t = 0

    logging.info(
        "*** state names %s \naction names %s",
        str(self.state_names),
        str(self.action_names),
    )

    while t < n_extra_steps:
      rng, rng_input = random.split(rng)

      action = agent.observe(rng_input, self.x, n_extra_steps - t)

      logging.info(
          "*** time %d, state %s, action %d", t, str(self.x), action
      )

      logging.info(
          "*** state %s, action %s",
          str(self.state_names[np.flatnonzero(self.x)]),
          self.action_names[action],
      )

      rng, rng_input = random.split(rng)
      rands_to_use = None
      if rands_list_to_use is not None:
        rands_to_use = rands_list_to_use[t]

      self.advance(rng_input, action, rands_to_use=rands_to_use)

      current_reward = self.current_reward()
      reward_sum += current_reward
      t += 1
      logging.info("time: %d, Accumulated reward: %f", t, reward_sum)

    return reward_sum
