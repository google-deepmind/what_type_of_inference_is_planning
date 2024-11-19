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

"""Abstract class for agent."""
import abc
import jax
import numpy as np


class Agent(abc.ABC):
  """Agent abstract class."""

  @abc.abstractmethod
  def observe(
      self, rng: jax.Array, x: np.ndarray, remaining_steps: int
  ) -> int:
    """Observe the environment.

    Args:
      rng: Random number generator key.
      x: Observation.
      remaining_steps: Remaining steps in the episode.

    Returns:
      Action chosen by the agent.
    """
