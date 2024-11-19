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

"""Value belief propagation algorithm."""
# pylint: disable=g-explicit-length-test  (for better readability)
# pylint: disable=unused-variable  (for better readability)
import warnings

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import tqdm

from what_type_of_inference_is_planning.agents.algorithms import updates


def order_transf(
    t: np.ndarray, dep: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Reorder the dependencies so that variables are in ascending order.

    Not currently used.

  Args:
    t: Transition matrices
    dep: dep[i] lists the variables that variable i depends on

  Returns:
    Reordered t and dep
  """
  t, dep = t.copy(), dep.copy()
  assert t.shape[0] == dep.shape[0]
  for i in range(len(t)):
    assert len(t[i].shape) - 2 == dep[i].shape[0]
    if dep[i].shape[0] > 0:
      order = dep[i].argsort()
      dep[i] = dep[i][order]
      t[i] = t[i].transpose(np.hstack((order, len(order) + np.arange(2))))
    assert len(np.unique(dep[i])) == len(dep[i])
  return t, dep


def connections(dep: list[np.ndarray], n_e: int) -> updates.CnxStructure:
  """Compute connectivity structure for the transitions.

  Args:
    dep: dep[i] lists the variables that variable i depends
      on in the DBN dynamics.
    n_e: Total number of entities

  Returns:
    Connectivity structure for the transitions
  """
  n_r = len(dep)
  c_mat = np.zeros((n_r, n_e))
  f_cnxns_mat, f_cnxns_mat2 = [], []
  for i in range(n_r):
    if len(dep[i]) > 0:
      c_mat[i, dep[i]] = 1
    f = tuple(dep[i])
    expand = tuple(tuple(a) for a in 1 - 2 * np.eye(len(f), dtype=int))
    d = len(f)
    if d > 0:
      m_mat = (np.ones((d, 1), int) * np.arange(d))[np.eye(d) == 0].reshape(
          d, d - 1
      )
      compress = tuple(tuple(m) for m in m_mat)
    else:
      compress = ()
    f_cnxns_mat.append((f, expand, compress))
  b_cnxns_mat = []
  for i in range(n_e):
    b = tuple(c_mat[:, i].nonzero()[0])
    idx = tuple((np.array(f_cnxns_mat[v][0]) == i).argmax() for v in b)
    b_cnxns_mat.append((b, idx))
  for i in range(n_r):
    f, expand, compress = f_cnxns_mat[i]
    idx = tuple((np.array(b_cnxns_mat[v][0]) == i).argmax() for v in f)
    f_cnxns_mat2.append((f, idx, expand, compress))
  cnxnt_mat = tuple([tuple(np.arange(len(dep[i]))) for i in range(len(dep))])
  return tuple(f_cnxns_mat2), tuple(b_cnxns_mat), cnxnt_mat


def vbp(
    rng: jax.Array,
    start: np.ndarray,
    transf: list[np.ndarray],
    transf_dep: list[np.ndarray],
    reward: list[np.ndarray],
    reward_dep: list[np.ndarray],
    n_steps: int,
    cvx: bool = False,
    from_qxa: bool = True,
    reward_allt: bool = True,
    max_iter: int = 100,
    damping: float = 0.5,
    mess_fwd: jax.Array | None = None,
    min_eps: float = 1e-2,
    tol: float = 1e-2,
    init_n: float = 1.0,
    min_val: float = -1e3,
    min_reward: float = 1e-2,
    max_inner_iter: int = 300,
    progress_bar: bool = False,
    additive_reward: bool = True,
    lambd: float = 1.0,
) -> tuple[
    jax.Array,
    np.ndarray,
    np.ndarray,
    jax.Array,
    jax.Array,
    np.ndarray,
    np.ndarray,
]:
  """Runs value belief propagation (VBP).

  Args:
    rng: A a PRNG key
    start: Initial state of each variable in the factored MDP
    transf: Transition matrices for the dynamics
    transf_dep: transf_dep[i] lists the variables that variable i depends
      on in the DBN dynamics.
    reward: Analogous to the transition matrices for the rewards
    reward_dep: reward_dep[r] lists the variables that reward r depends on
    n_steps: Number of steps to unroll the factored MDP for
    cvx: Whether to use the weighting numbers from the convex approximation
    from_qxa: Whether to derive the message updates through qxa (numerically
      different)
    reward_allt: score reward at all time steps instead of just the last one
    max_iter: Maximum number of VBP iterations
    damping: VBP damping factor (0.0 ... 1.0, with 1.0 being no damping).
    mess_fwd: Initial values for the forward messages
    min_eps: Minimum value for epsilon (see paper)
    tol: Tolerance for the detection of VBP having converged
    init_n: Level of noise to initialize the forward messages
    min_val: Minimum value that messages can have (log-space)
    min_reward: Minimum value for the transformed reward (see paper)
    max_inner_iter: Maximum number of iterations to solve a time slice
    progress_bar: Whether to visualize a progress bar
    additive_reward: Whether rewards should be treated as additive
      (instead of multiplicative)
    lambd: Scaling factor for the rewards (see paper)

  Returns:
    Tuple of
      Approximate total reward
      Untransformed rewards at each timestep
      logarithm of the action pseudomarginals
      Forward messages
      Backward messages
      Evolution of the deltas along VBP iterations, to assess convergence
      Evolution of the best action along VBP iterations
  """
  n_e = len(transf)
  n_r = len(reward)
  n_a, n_s = transf[0].shape[-2:]
  assert min_val < 0
  assert min_reward > 0
  assert start.shape == (n_e,)
  assert len(transf_dep) == n_e
  assert len(reward_dep) == n_r

  t_cnxns = connections(transf_dep, n_e)
  r_cnxns = connections(reward_dep, n_e)
  logt_mat = [jnp.log(transf[i]) for i in range(n_e)]

  if additive_reward:
    r_shift = min([reward[r].min() for r in range(n_r)]) - min_reward
    logr_mat = [jnp.log(reward[i] - r_shift) for i in range(n_r)]
  else:
    r_shift = 0.0
    logr_mat = [lambd * reward[i] for i in range(n_r)]

  if mess_fwd is None:
    rng, rng_input = random.split(rng)
    mess_fwd = init_n * random.gumbel(rng_input, (n_steps, n_e, n_s))
    # with jnp.printoptions(precision=30, suppress=True):
    #   logging.info("*** mess_fwd\n %s", str(mess_fwd))
  mess_bwd = jnp.zeros((n_steps, n_e, n_s))

  # f_matix starting state and normalize last state
  mess_fwd = mess_fwd.at[0].set(min_val).at[0, jnp.arange(n_e), start].set(0)
  mess_fwd = mess_fwd.at[-1].set(jax.nn.log_softmax(mess_fwd[-1], axis=1))

  def step(t, mess_fwd, mess_bwd, eps):
    f_mat, log_qxa, fwd_out, bwd_out, delta_, iter_ = updates.vbp1step(
        mess_fwd[t],
        mess_bwd[t + 1],
        logt_mat,
        t_cnxns,
        cvx,
        eps,
        min_val,
        from_qxa,
        max_inner_iter,
        tol,
    )

    if delta_ > tol:
      warnings.warn(f"Number of iteration exceeded, delta {delta_} > tol {tol}")
    if reward_allt:
      if additive_reward:
        f_mat, bwd_out = updates.bwd_reward(
            mess_fwd[t],
            bwd_out,
            logr_mat,
            r_cnxns,
            f_mat,
            min_val,
            additive_reward,
        )
      else:
        mess_to_reward = jax.nn.log_softmax(mess_fwd[t] + bwd_out, axis=1)
        f_mat, bwd_out = updates.bwd_reward(
            mess_to_reward,
            bwd_out,
            logr_mat,
            r_cnxns,
            f_mat,
            min_val,
            additive_reward,
        )
        mess_to_reward = jax.nn.log_softmax(fwd_out + mess_bwd[t + 1], axis=1)
        _, fwd_out = updates.bwd_reward(
            mess_to_reward,
            fwd_out,
            logr_mat,
            r_cnxns,
            0.0,
            min_val,
            additive_reward,
        )
        fwd_out = jax.nn.log_softmax(fwd_out, axis=1)
    else:
      f_mat, bwd_out = updates.bwd_reward(
          mess_fwd[t],
          bwd_out,
          logr_mat[:0],
          r_cnxns,
          f_mat,
          min_val,
          additive_reward,
      )
    return f_mat, fwd_out, bwd_out, iter_, log_qxa

  # Main loop
  convergence = []
  actions = []
  if additive_reward:
    null_mess = jnp.zeros((n_e, n_s))
    f_matfut = -jnp.inf
  else:
    null_mess = jnp.ones((n_e, n_s))
    f_matfut = 0.0

  f_mats = np.zeros(n_steps)
  tinit = 0
  logq_mata = np.zeros((n_steps - 1, n_a))
  if progress_bar:
    pbar = tqdm.trange(max_iter)
  else:
    pbar = range(max_iter)

  log_qxa_all = None
  for it in pbar:
    delta_sum = 0.0
    accel = 2.0 / min_eps / max_iter
    eps = jax.lax.max(min_eps, 1.0 / (accel * it + 1))

    t = n_steps - 1
    f_mats[t], mess_bwd_ = updates.bwd_reward(
        mess_fwd[t],
        null_mess,
        logr_mat,
        r_cnxns,
        f_matfut,
        min_val,
        additive_reward,
    )
    delta_sum += jnp.abs(mess_bwd_ - mess_bwd[t]).sum()
    mess_bwd = mess_bwd.at[t].set(mess_bwd_)
    for t in range(n_steps - 2, -1, -1):
      f_mats[t], fwd_out, bwd_out, iter_, log_qxa = step(
          t, mess_fwd, mess_bwd, eps
      )
      delta = bwd_out - mess_bwd[t]
      delta_sum += jnp.abs(delta).sum()
      mess_bwd = mess_bwd.at[t].add(damping * delta)
      tinit += iter_
    log_qxa_all = []
    for t in range(n_steps - 1):
      _, fwd_out, bwd_out, iter_, log_qxa = step(t, mess_fwd, mess_bwd, eps)
      log_qxa_all.append(log_qxa)
      delta = fwd_out - mess_fwd[t + 1]
      delta_sum += jnp.abs(delta).sum()
      mess_fwd = mess_fwd.at[t + 1].add(damping * delta)
      tinit += iter_
      log_qa_unn = jax.nn.logsumexp(log_qxa[0].reshape(-1, n_a), axis=0)
      logq_mata[t] = jax.nn.log_softmax(log_qa_unn)
      d = len(log_qxa[0].shape)
      logq_mata[t] = jax.nn.log_softmax(
          jax.nn.logsumexp(log_qxa[0], axis=tuple(jnp.arange(d - 1)))
      )
    if additive_reward:
      score = jnp.exp(f_mats[0]) + r_shift * n_r * (
          n_steps if reward_allt else 1
      )
    else:
      score = f_mats[0] / lambd
    p = logq_mata[0] == logq_mata[0].max()
    p = p / p.sum()
    rng, rng_input = random.split(rng)
    actions.append(random.categorical(rng_input, jnp.log(p)))

    convergence.append(score)

    if delta_sum < tol:
      break

  score_l = r_shift + jnp.exp(
      score_last(
          mess_fwd=mess_fwd,
          mess_bwd=mess_bwd,
          log_qxa_all=log_qxa_all,
          logt_mat=logt_mat,
          cvx=False,
          r_cnxns=r_cnxns,
          logr_mat=logr_mat,
      )
  )
  return (
      score_l,
      f_mats,
      logq_mata,
      mess_fwd,
      mess_bwd,
      np.array(convergence),
      np.array(actions),
  )


def score_last(
    mess_fwd: jax.Array,
    mess_bwd: jax.Array,
    log_qxa_all: list[list[jax.Array]],
    logt_mat: list[jax.Array],
    cvx: bool,
    r_cnxns: updates.CnxStructure,
    logr_mat: list[jax.Array],
) -> jax.Array:
  """Score the reward obtained at the last timestep only.

  Args:
    mess_fwd: Forward messages
    mess_bwd: Backward messages
    log_qxa_all: Log of pseudomarginals at all xa factors
    logt_mat: Log of transition matrices for the dynamics
    cvx: Whether to use the weighting numbers from the convex approximation
    r_cnxns: Conectivity structure for the rewards
    logr_mat: Log of the analogous to the transition matrices for the rewards

  Returns:
    Untransformed score at the last time step only
  """
  n_steps = mess_bwd.shape[0]
  n_e = len(logt_mat)
  f_mat = jnp.array(0.0)
  for t in range(n_steps - 1):
    bwd_in = mess_bwd[t + 1]
    for e in range(n_e):
      d = len(logt_mat[e].shape)
      log_qxp_xa = jax.nn.log_softmax(
          logt_mat[e] + bwd_in[e][(None,) * (d - 1)], axis=d - 1
      )
      log_qxaxp = jax.nn.log_softmax(
          log_qxa_all[t][e][..., None] + log_qxp_xa, axis=None
      )
      # Energy of transition + H(x'|x,a)
      f_mat += (
          jnp.exp(log_qxaxp)
          * (logt_mat[e].clip(-1e38, jnp.inf) - log_qxp_xa.clip(-1e38, jnp.inf))
      ).sum()

      if not cvx and d > 2:
        log_qx = jax.nn.logsumexp(log_qxa_all[t][e], axis=-1)
        f_mat += -(jnp.exp(log_qx) * log_qx).sum()
        for i in range(d - 2):
          axis = tuple(j for j in range(d - 2) if j != i)
          log_qxv = jax.nn.logsumexp(log_qx, axis=axis)
          f_mat += (jnp.exp(log_qxv) * log_qxv).sum()

  # Reward at the end
  n_r = len(logr_mat)
  n_s = logt_mat[0].shape[-1]
  f_cnxns_mat = r_cnxns[0]
  fwd_in = mess_fwd[n_steps - 1]
  fwd = None
  for r in range(n_r):
    f_cnxn, idx, expand, compress = f_cnxns_mat[r]
    assert f_cnxn  # Reward must have parents
    n_v = len(f_cnxn)
    for var in f_cnxn:
      fwd = jnp.zeros((n_s,) * n_v)
      for i, v in enumerate(f_cnxn):
        fwd += fwd_in[v].reshape(expand[i])
    log_q_r_mat = jax.nn.log_softmax(fwd + logr_mat[r], axis=None)
    # Expectation of reward minus entropy
    f_mat += (jnp.exp(log_q_r_mat) * (logr_mat[r] - log_q_r_mat)).sum()
    for i in range(n_v):
      axis = tuple(j for j in range(n_v) if j != i)
      log_q_r_matv = jax.nn.logsumexp(log_q_r_mat, axis=axis)
      f_mat += (jnp.exp(log_q_r_matv) * log_q_r_matv).sum()
  return f_mat
