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

"""Maximum Entropy Value Belief Propagation (MEVBP)."""

import functools
import warnings

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import tqdm


CnxStructure = tuple[
    tuple[
        tuple[
            tuple[int, ...],
            tuple[int, ...],
            tuple[tuple[int, ...], ...],
            tuple[tuple[int, ...], ...],
        ],
        ...,
    ],
    tuple[tuple[tuple[int], tuple[int]]],
    tuple[tuple[int]],
]


def connections(dep: list[np.ndarray], n_e: int) -> CnxStructure:
  """Compute connectivity structure for the transitions.

  Args:
    dep: dep[i] lists the variables that variable i depends on in the DBN
      dynamics.
    n_e: Total number of entities

  Returns:
    Connectivity structure for the transitions
  """
  n_r = len(dep)
  c_mat = np.zeros((n_r, n_e))
  f_cnxns_mat, f_cnxns_mat2 = [], []
  for i in range(n_r):
    if dep[i].size > 0:
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
    idx = tuple((np.array(b_cnxns_mat[v][0]) == i).argmax() for v in f)  # pytype: disable=name-error
    f_cnxns_mat2.append((f, idx, expand, compress))
  cnxnt_mat = tuple([tuple(np.arange(len(dep[i]))) for i in range(len(dep))])
  return tuple(f_cnxns_mat2), tuple(b_cnxns_mat), cnxnt_mat


@functools.partial(jax.jit, static_argnames="t_cnxns")
def create_joints(fwd_in: jax.Array, t_cnxns: CnxStructure) -> list[jax.Array]:
  """Creates a joint fwd message from the marginals per variable.

  Args:
    fwd_in: forward messages per variable in log space
    t_cnxns: Connectivity structure for the dynamics

  Returns:
    Joint forward message
  """
  n_s = fwd_in.shape[1]
  f_cnxns_mat, _, _ = t_cnxns
  mess_jx_f_inc_all = []
  for e in range(len(f_cnxns_mat)):
    # Update incoming messages
    # Forward
    f_cnxn, _, expand, _ = f_cnxns_mat[e]
    if f_cnxn:  # If this entity has any parents
      n_v = len(f_cnxn)
      assert n_v > 0
      mess_jx_f_inc = jnp.zeros((n_s,) * n_v)
      for i, v in enumerate(f_cnxn):
        mess_jx_f_inc += fwd_in[v].reshape(expand[i])
    else:
      mess_jx_f_inc = jnp.zeros(1)
    mess_jx_f_inc_all.append(mess_jx_f_inc)
  return mess_jx_f_inc_all


@jax.jit
def update(
    mess_jx_b_out: list[jax.Array],
    mess_a_inc: jax.Array,
    mess_jx_f_inc: jax.Array,
    mess_xa_b_inc: list[jax.Array],
) -> tuple[list[jax.Array], jax.Array]:
  """Compute updated backward messages and incoming to action messages.

  Args:
    mess_jx_b_out: Backward outgoing messages from joint parent variables
    mess_a_inc: Incoming messages to the action variable
    mess_jx_f_inc: Forward incoming messages to joint parent variables
    mess_xa_b_inc: Factor between the joint parent variables and the action
      after the backwards incoming messages has been marginalized

  Returns:
    Tuple of
      Backward outgoing messages from joint parent variables
      Incoming messages to the action variable
  """
  n_e = len(mess_jx_b_out)
  bel_a = mess_a_inc.sum(0)
  for e in range(n_e):
    mess_a_out = bel_a - mess_a_inc[e]
    mess_jx_b_out[e] = jax.nn.logsumexp(
        mess_xa_b_inc[e] + mess_a_out[None], axis=1
    )
    mess_a_inc_new = jax.nn.logsumexp(
        (mess_jx_f_inc[e] - mess_jx_b_out[e])[:, None] + mess_xa_b_inc[e],
        axis=0,
    )
    bel_a += mess_a_inc_new - mess_a_inc[e]
    mess_a_inc = mess_a_inc.at[e].set(mess_a_inc_new)
  return mess_jx_b_out, mess_a_inc


@functools.partial(jax.jit, static_argnums=(3, 4))
def solve1step(
    mess_jx_f_inc: jax.Array,
    tr_mat: list[jax.Array],
    bwd_in: jax.Array,
    max_inner_iter: int,
    tol: float,
) -> tuple[float, list[jax.Array], jax.Array, jax.Array, jax.Array, int, float]:
  """Solve the optimization problem corresponding to 1 step of the MDP.

  Args:
    mess_jx_f_inc: Forward incoming messages to joint parent variables
    tr_mat: Transition matrices for the dynamics
    bwd_in: Backward incoming message into the "back" of the slice
    max_inner_iter: Number of iterations that the solver can run for
    tol: Required tolerance for solver termination

  Returns:
    Tuple of
      Estimated value for the slice
      Backward outgoing messages from joint parent variables
      Forward outgoing message out of the "back" of the slice
      log of the pseudomarginals at xa factors
      log of the pseudomarginals at the action variable
      Number of iterations that were run
      Message delta in the last iteration
  """
  n_e, n_a = len(tr_mat), tr_mat[0].shape[-2]
  mess_xa_b_inc = [
      (tr_mat[e] * bwd_in[e, None, None]).sum(2) for e in range(n_e)
  ]

  mess_a_inc = jnp.zeros((n_e, n_a))
  mess_jx_b_out = [jnp.zeros_like(mess_jx_f_inc[e]) for e in range(n_e)]

  def step(carry):
    mess, _, n_iter = carry
    mess_jx_b_out, mess_a_inc = mess
    mess_jx_b_out, mess_a_inc_new = update(
        mess_jx_b_out, mess_a_inc, mess_jx_f_inc, mess_xa_b_inc
    )
    delta = jnp.abs(mess_a_inc_new - mess_a_inc).sum()
    mess = mess_jx_b_out, mess_a_inc_new
    return mess, delta, n_iter + 1

  mess = mess_jx_b_out, mess_a_inc
  mess, delta, n_iter = jax.lax.while_loop(
      lambda carry: (carry[2] < max_inner_iter) & ((carry[1] > tol)),
      step,
      (mess, jnp.inf, 0),
  )
  mess_jx_b_out, mess_a_inc = mess
  mess_a_out = mess_a_inc.sum(0, keepdims=True) - mess_a_inc

  log_qxa = [
      (mess_jx_f_inc[e] - mess_jx_b_out[e])[:, None]
      + mess_xa_b_inc[e]
      + mess_a_out[e, None]
      for e in range(n_e)
  ]
  log_qa = jnp.array([jax.nn.logsumexp(log_qxa[e], axis=0) for e in range(n_e)])
  log_qa = jax.nn.logsumexp(log_qa, axis=0) - jnp.log(n_e)

  fwd_out = jnp.log(
      jnp.array([
          (jnp.exp(log_qxa[e])[:, :, None] * tr_mat[e]).sum((0, 1))
          for e in range(n_e)
      ])
  )

  val = jnp.array(
      [jnp.exp(mess_jx_f_inc[e]) @ mess_jx_b_out[e] for e in range(n_e)]
  ).sum()
  log_qxa = None  # to save memory since this is only for debugging
  return val, mess_jx_b_out, fwd_out, log_qxa, log_qa, n_iter, delta


@functools.partial(jax.jit, static_argnums=(3, 4, 5))
def maxent_vbp1step(
    fwd_in: jax.Array,
    bwd_in: jax.Array,
    tr_mat: list[jax.Array],
    t_cnxns: CnxStructure,
    max_inner_iter: int,
    tol: float,
) -> tuple[float, jax.Array, jax.Array, jax.Array, jax.Array, int, float]:
  """Run one step of MaxEnt VBP.

  Args:
    fwd_in: Forward incoming message into the "front" of the slice
    bwd_in: Backward incoming message into the "back" of the slice
    tr_mat: Transition matrices for the dynamics
    t_cnxns: Connectivity structure for the dynamics
    max_inner_iter: Number of iterations that the solver can run for
    tol: Required tolerance for solver termination

  Returns:
    Tuple of
      Estimated value for the slice
      Forward outgoing message out of the "back" of the slice
      Backward outgoing messages out of the "front" of the slice
      log of the pseudomarginals at xa factors
      log of the pseudomarginals at the action variable
      Number of iterations that were run
      Message delta in the last iteration
  """
  n_e = len(fwd_in)

  mess_jx_f_inc = create_joints(fwd_in, t_cnxns)
  mess_jx_f_inc_flat = [m.reshape(-1) for m in mess_jx_f_inc]

  val, mess_jx_b_out_flat, fwd_out, log_qxa, log_qa, n_iter, delta = solve1step(
      mess_jx_f_inc_flat, tr_mat, bwd_in, max_inner_iter, tol
  )

  mess_jx_b_out = [
      mess_jx_b_out_flat[e].reshape(mess_jx_f_inc[e].shape) for e in range(n_e)
  ]
  bwd_out = backward(fwd_in, mess_jx_b_out, t_cnxns)
  bwd_out -= (jnp.exp(fwd_in) * bwd_out).sum(1, keepdims=True)
  return val, fwd_out, bwd_out, log_qxa, log_qa, n_iter, delta


@functools.partial(jax.jit, static_argnames="r_cnxns")
def backward(
    fwd_in: jax.Array, rew_scaled: list[jax.Array], r_cnxns: CnxStructure
) -> jax.Array:
  """Computes the backward messages from the rewards and input messsages.

  Args:
    fwd_in: Forward incoming message into the "front" of the slice
    rew_scaled: Scaled version of the rewards
    r_cnxns: Connectivity structure for the reward

  Returns:
    Backward outgoing messages from joint parent variables
  """
  n_e, n_s = fwd_in.shape

  n_r = len(rew_scaled)
  assert len(r_cnxns) == 3

  # Handling rewards
  r_bwd = jnp.zeros((n_e, n_r, n_s))
  f_cnxn_mat = r_cnxns[0]
  for r in range(n_r):
    f_cnxn, _, expand, _ = f_cnxn_mat[r]
    if f_cnxn:  # if it has parents
      n_v = len(f_cnxn)
      for var in f_cnxn:
        fwd = jnp.zeros((n_s,) * n_v)
        axis = ()
        for i, v in enumerate(f_cnxn):
          if v == var:
            continue
          fwd += fwd_in[v].reshape(expand[i])
          axis = axis + (i,)
        r_bwd = r_bwd.at[var, r].set(
            (jnp.exp(fwd) * rew_scaled[r]).sum(axis=axis)
        )
  return r_bwd.sum(1)


def mevbp(
    rng: jax.Array,
    start: np.ndarray,
    transf: list[np.ndarray],
    transf_dep: list[np.ndarray],
    reward: list[np.ndarray],
    reward_dep: list[np.ndarray],
    n_steps: int,
    max_iter: int = 100,
    damping: float = 0.5,
    alpha: float = 1.0,
    mess_fwd: jax.Array | None = None,
    tol: float = 1e-2,
    init_n: float = 1.0,
    min_val: float = -1e3,
    max_inner_iter: int = 300,
    progress_bar: bool = False,
    rescale_reward: bool = True,
) -> tuple[
    float,
    np.ndarray,
    jax.Array,
    jax.Array,
    jax.Array,
    np.ndarray,
    np.ndarray,
    jax.Array,
    float,
]:
  """Runs MaxEnt value belief propagation (MaxEnt VBP).

  Args:
    rng: A a PRNG key
    start: Initial state of each variable in the factored MDP
    transf: Transition matrices for the dynamics
    transf_dep: transf_dep[i] lists the variables that variable i depends on in
      the DBN dynamics.
    reward: Analogous to the transition matrices for the rewards
    reward_dep: reward_dep[r] lists the variables that reward r depends on
    n_steps: Number of steps to unroll the factored MDP for
    max_iter: Maximum number of MaxEnt VBP iterations
    damping: MaxEnt VBP damping factor (0.0 ... 1.0, with 1.0 being no damping)
    alpha: scale of the policy regularization (see paper)
    mess_fwd: Initial values for the forward messages
    tol: Tolerance for the detection of MaxEnt VBP having converged
    init_n: Level of noise to initialize the forward messages
    min_val: Minimum value that messages can have (log-space)
    max_inner_iter: Maximum number of iterations to solve a time slice
    progress_bar: Whether to visualize a progress bar
    rescale_reward: Whether to rescale the reward to unit range.

  Returns:
    Tuple of
      Approximate total reward
      Untransformed rewards at each timestep
      logarithm of the action pseudomarginals
      Forward messages
      Backward messages
      Evolution of the scores along MaxEnt VBP iterations
      Evolution of the best action along MaxEnt VBP iterations
      Logarithm of the xa pseudomarginals
      Absolute message change in the last MaxEnt VBP iteration
  """
  n_e = len(transf)
  n_r = len(reward)
  n_a, n_s = transf[0].shape[-2:]
  assert min_val < 0
  assert start.shape == (n_e,)
  assert len(transf_dep) == n_e
  assert len(reward_dep) == n_r

  t_cnxns = connections(transf_dep, n_e)
  r_cnxns = connections(reward_dep, n_e)
  tr_mat = [np.array(transf[i]).reshape(-1, n_a, n_s) for i in range(n_e)]
  norm_r = alpha * (
      max([np.array(r).max() - np.array(r).min() for r in reward])
      if rescale_reward
      else 1.0
  )
  rew_scaled = [reward[i] / norm_r for i in range(n_r)]

  if mess_fwd is None:
    rng, rng_input = random.split(rng)
    mess_fwd = init_n * random.gumbel(rng_input, (n_steps, n_e, n_s))
  else:
    assert mess_fwd.shape == (n_steps, n_e, n_s)
  mess_fwd = jax.nn.log_softmax(mess_fwd.clip(min_val, 0), axis=2)
  mess_bwd = jnp.zeros((n_steps, n_e, n_s))

  # Fix starting state and normalize all time steps
  mess_fwd = mess_fwd.at[0].set(min_val).at[0, jnp.arange(n_e), start].set(0)
  mess_fwd = jax.nn.log_softmax(mess_fwd, axis=2)

  def step(t, mess_fwd, mess_bwd):
    val, fwd_out, bwd_out, log_qxa, log_qa, n_iter, delta = maxent_vbp1step(
        mess_fwd[t], mess_bwd[t + 1], tr_mat, t_cnxns, max_inner_iter, tol
    )
    if n_iter == max_inner_iter:
      warnings.warn(
          f"""Maximum number of inner iterations {max_inner_iter} reached.
             Delta was {delta}."""
      )
    bwd_out_r = backward(mess_fwd[t], rew_scaled, r_cnxns)
    bwd_out += bwd_out_r
    return val, fwd_out, bwd_out, log_qxa, log_qa, n_iter

  # Main loop
  convergence = []
  actions = []
  val_step = np.zeros(n_steps)
  bwd_out = backward(mess_fwd[0], rew_scaled, r_cnxns)
  val_step[0] = (jnp.exp(mess_fwd[0]) * bwd_out).sum()
  log_qa_all = np.zeros((n_steps - 1, n_a))
  pbar = tqdm.tqdm(range(max_iter), disable=not progress_bar)
  n_iter, delta_sum, log_qxa_all = 0, 0.0, None
  for _ in pbar:
    delta_sum = 0.0

    t = n_steps - 1
    bwd_out = backward(mess_fwd[t], rew_scaled, r_cnxns)
    # delta_sum += jnp.abs(bwd_out - mess_bwd[t]).sum()
    mess_bwd = mess_bwd.at[t].set(bwd_out)

    for t in range(n_steps - 2, -1, -1):
      val_step[t + 1], _, bwd_out, _, _, _ = step(t, mess_fwd, mess_bwd)
      delta = bwd_out - mess_bwd[t]
      # delta_sum += jnp.abs(delta).sum()
      mess_bwd = mess_bwd.at[t].add(damping * delta)
    log_qxa_all = []
    for t in range(n_steps - 1):
      _, fwd_out, _, log_qxa, log_qa, n_iter = step(t, mess_fwd, mess_bwd)
      fwd_out = fwd_out.clip(min_val, 0)
      delta = fwd_out - mess_fwd[t + 1]
      delta_sum += jnp.abs(delta).sum()
      mess_fwd = mess_fwd.at[t + 1].add(damping * delta)
      mess_fwd = mess_fwd.at[t + 1].set(
          jax.nn.log_softmax(mess_fwd[t + 1], axis=1).clip(min_val, 0)
      )
      log_qa_all[t] = log_qa
      log_qxa_all.append(log_qxa)
    val_step *= norm_r
    score = val_step.sum()

    p = log_qa_all[0] == log_qa_all[0].max()
    log_p = jnp.log(p / p.sum())
    rng, rng_input = random.split(rng)
    actions.append(random.categorical(rng_input, log_p))

    convergence.append(score)
    pbar.set_postfix(
        delta=delta_sum,
        score=score,
        a=actions[-1],
        estim_r=norm_r * score_all(mess_fwd, rew_scaled, r_cnxns),
        inner_iter=n_iter,
    )
    if delta_sum < tol:
      break

  return (
      norm_r * score_all(mess_fwd, rew_scaled, r_cnxns),
      val_step,
      log_qa_all,
      mess_fwd,
      mess_bwd,
      np.array(convergence),
      np.array(actions),
      log_qxa_all,
      delta_sum,
  )


def score_all(
    mess_fwd: jax.Array, rew_scaled: list[jax.Array], r_cnxns: CnxStructure
) -> float:
  """Computes the score of the current messages.

  Args:
    mess_fwd: Forward incoming message into the "front" of the slice
    rew_scaled: Scaled version of the rewards
    r_cnxns: Connectivity structure for the reward

  Returns:
    Score of the current messages
  """
  n_steps = mess_fwd.shape[0]
  n_r = len(rew_scaled)
  score = 0.0
  for t in range(n_steps):
    log_fwd = create_joints(mess_fwd[t], r_cnxns)
    for r in range(n_r):
      score += (jnp.exp(log_fwd[r]) * rew_scaled[r]).sum()
  return score
