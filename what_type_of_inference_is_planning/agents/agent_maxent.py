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

"""MaxEnt VBP agent."""

from absl import flags
from absl import logging
import jax
from jax import random
import numpy as np

from what_type_of_inference_is_planning.agents import agent_base
from what_type_of_inference_is_planning.agents.algorithms import maxent_vbp
from what_type_of_inference_is_planning.envs import environment

_MAX_RETRIES = flags.DEFINE_integer(
    "max_retries", default=5, help="maximum number of times to rerun the solver"
)
_MAX_ITER = flags.DEFINE_integer(
    "max_iter", default=500, help="max outer iterations for solver"
)
_MAX_INNER_ITER = flags.DEFINE_integer(
    "max_inner_iter", default=10_000, help="max inner iterations for solver"
)
_DAMPING = flags.DEFINE_float("damping", default=0.5, help="Damping for solver")
_ALPHA = flags.DEFINE_float(
    "alpha", default=0.1, help="Scale of the policy regularization"
)


class AgentME(agent_base.Agent):
  """MaxEnt Value Belief Propagation (VBP) agent.

  Attributes:
    env: The environment.
    horizon: The horizon length.
  """

  def __init__(self, env: environment.Environment, horizon: int):
    self.env = env
    self.horizon = horizon

  def observe(self, rng: jax.Array, x: np.ndarray, remaining_steps: int) -> int:
    """Observe the environment.

    Args:
      rng: Random number.
      x: Current state.
      remaining_steps: Number of remaining steps.

    Returns:
      action: Action chosen.
    """
    delta_sum, count = np.inf, 0
    max_retries = _MAX_RETRIES.value
    deltas, actions = [], []
    log_qa_all, convergence = [0], []
    while delta_sum > 1e-2 and count < max_retries:
      count += 1
      rng, rng_input = random.split(rng)
      n_steps = min(self.horizon, remaining_steps + 1)
      (
          _,
          _,
          log_qa_all,
          _,
          _,
          convergence,
          a,
          _,
          delta_sum,
      ) = maxent_vbp.mevbp(
          rng_input,
          x,
          self.env.transition_prob,
          self.env.state_dependencies,
          self.env.reward,
          self.env.reward_dependencies,
          n_steps,
          max_iter=_MAX_ITER.value,
          damping=_DAMPING.value,
          alpha=_ALPHA.value,
          mess_fwd=None,
          tol=1e-2,
          init_n=1.0,
          min_val=-1e3,
          max_inner_iter=_MAX_INNER_ITER.value,
          progress_bar=False,
          rescale_reward=True,
      )
      deltas.append(delta_sum)
      actions.append(a[-1])

    # choose action of the run with the smallest delta
    action = min(zip(actions, deltas), key=lambda x: x[1])[0]

    with np.printoptions(precision=20, suppress=True):
      logging.info("*** prob = %s", str(np.exp(log_qa_all[0])))
      logging.info("*** convergence = %s", str(convergence[-20:]))
    logging.info("*** Action chosen: %d", action)

    return action
