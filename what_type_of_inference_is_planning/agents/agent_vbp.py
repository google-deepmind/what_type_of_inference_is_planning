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

"""VBP agent."""
from absl import flags
from absl import logging
import jax
from jax import numpy as jnp
from jax import random
import numpy as np

from what_type_of_inference_is_planning.agents import agent_base
from what_type_of_inference_is_planning.agents.algorithms import value_bp
from what_type_of_inference_is_planning.envs import environment


_DAMPING = flags.DEFINE_float(
    "damping", default=0.1, help="damping for solver"
)
_MAX_ITER = flags.DEFINE_integer(
    "max_iter", default=500, help="max outer iterations"
)
_MAX_INNER_ITER = flags.DEFINE_integer(
    "max_inner_iter", default=300, help="max inner iterations"
)
_LAMBDA = flags.DEFINE_float(
    "lambda", default=0.3, help="lambda for reward scaling"
)
_ADDITIVE_REWARD = flags.DEFINE_bool(
    "additive_reward",
    default=False,
    help=(
        "whether rewards should be treated as additive (instead of"
        " multiplicative)"
    ),
)


class AgentVBP(agent_base.Agent):
  """Value Belief Propagation (VBP) agent.

  Attributes:
    env: The environment.
    horizon: The horizon length.
    lambd_scaled: Scaling factor for the rewards (see paper).
  """

  def __init__(self, env: environment.Environment, horizon: int):
    self.env = env
    self.horizon = horizon

    max_ptp_r = max([r.ptp() for r in self.env.reward])
    lambd = _LAMBDA.value
    self.lambd_scaled = lambd / max_ptp_r

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
    n_steps = min(self.horizon, remaining_steps + 1)
    _, _, logq_mata, _, _, convergence, actions = value_bp.vbp(
        rng_input,
        x,
        self.env.transition_prob,
        self.env.state_dependencies,
        self.env.reward,
        self.env.reward_dependencies,
        n_steps,
        cvx=False,
        reward_allt=True,
        max_iter=_MAX_ITER.value,
        damping=_DAMPING.value,
        mess_fwd=None,
        min_eps=1e-2,
        tol=1e-2,
        init_n=1.0,
        min_val=-1e3,
        min_reward=1e-2,
        max_inner_iter=_MAX_INNER_ITER.value,
        additive_reward=_ADDITIVE_REWARD.value,
        lambd=self.lambd_scaled,
    )
    reward = logq_mata[-1]
    p = reward == reward.max()
    p = p / p.sum()

    action = actions[-10:][convergence[-10:].argmax()]
    with np.printoptions(precision=20, suppress=True):
      logging.info("*** prob = %s, R = %s", str(jnp.log(p)), str(reward))
      logging.info("*** convergence = %s", str(convergence[-20:]))
      logging.info("*** all aciton = %s", str(actions[-20:]))
    logging.info("*** Action chosen: %d", action)

    return int(action)
