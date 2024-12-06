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

"""Message updates for value belief propagation."""
# pylint: disable=g-explicit-length-test  (for better readability)
# pylint: disable=unused-variable  (for better readability)
import functools
from typing import Any
import jax
import jax.numpy as jnp

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


@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def logsumexp_e(
    a_mat: jax.Array,
    eps: float,
    axis: tuple[int, ...] | None = None,
    keepdims: bool = False,
    fz: bool = False,
) -> jax.Array:
  """Compute logsumexp(a_mat/eps), with sum along specified axes.

  Args:
    a_mat: Matrix to which the transformation is applied
    eps: Current epsilon value (see paper)
    axis: Axes along which the sum takes place
    keepdims: Whether to keep dimensions along which the sum takes place
    fz: Whether to take the limit eps -> 0

  Returns:
    logsumexp(a_mat/eps), with sum along specified axes
  """
  # assert eps > 0, "Invalid temperature"
  a_matnorm = a_mat.max(axis=axis, keepdims=True)
  if fz:
    return a_matnorm.max(axis=axis, keepdims=keepdims)
  else:
    a_matnorm = a_matnorm.clip(-1e38, 1e38)
    return eps * jax.nn.logsumexp(
        (a_mat - a_matnorm) / eps, axis=axis, keepdims=keepdims
    ) + a_matnorm.max(axis=axis, keepdims=keepdims)


@functools.partial(jax.jit, static_argnums=(2, 3))
def logsoftmax_e(
    a_mat: jax.Array,
    eps: float,
    axis: tuple[int, ...] | None = None,
    fz: bool = False,
) -> jax.Array:
  """Compute logsoftmax(a_mat/eps), with softmax along specified axes.

  Args:
    a_mat: Matrix to which the transformation is applied
    eps: Current epsilon value (see paper)
    axis: Axes along which the softmax takes place
    fz: Whether to take the limit eps -> 0

  Returns:
    logsoftmax(a_mat/eps), with softmax along specified axes
  """
  # assert eps > 0, "Invalid temperature"
  if fz:
    return jax.nn.log_softmax(
        jnp.log(a_mat == a_mat.max(axis=axis, keepdims=True)), axis=axis
    )
  else:
    a_matnorm = a_mat.max(axis=axis, keepdims=True)
    return jax.nn.log_softmax((a_mat - a_matnorm) / eps, axis=axis)


@functools.partial(jax.jit, static_argnames=("r_cnxns", "additive_reward"))
def bwd_reward(
    fwd_in: jax.Array,
    bwd_fut: jax.Array,
    logr_mat: list[jax.Array],
    r_cnxns: CnxStructure,
    f_matfut: jax.Array,
    min_val: float,
    additive_reward: bool,
) -> tuple[jax.Array, Any]:
  """VBP message updates from the reward factors.

  Args:
    fwd_in: Forward incoming message into the "front" of the slice
    bwd_fut: Backward outgoing message out of the "front" of the next slice
    logr_mat: Log of the analogous to the transition matrices for the rewards
    r_cnxns: Connectivity structure for the rewards
    f_matfut: Estimated, untransformed value (combined reward from all future
      steps), from the next slice onwards
    min_val: Minimum value that messages can have (log-space)
    additive_reward: Whether rewards should be treated as additive
      (instead of multiplicative)

  Returns:
    Tuple of
      Estimated, untransformed value (combined reward from all future steps)
      Backward outgoing message out of the "front" of the current slice
  """
  n_e, n_s = fwd_in.shape

  n_r = len(logr_mat)
  assert len(r_cnxns) == 3
  assert (len(logr_mat) == 0) or (logr_mat[0].shape[0] == n_s)

  ## Handling rewards
  f_mat = jnp.zeros(n_r + 1)  # rewards current timestep + future
  r_bwd = jnp.zeros((n_e, n_r + 1, n_s))
  f_cnxns_mat = r_cnxns[0]
  for r in range(n_r):
    f_cnxn, idx, expand, compress = f_cnxns_mat[r]
    assert f_cnxn  # Reward must have parents
    n_v = len(f_cnxn)
    var = None
    for var in f_cnxn:
      fwd = jnp.zeros((n_s,) * n_v)
      axis = ()
      for i, v in enumerate(f_cnxn):
        if v == var:
          continue
        fwd += fwd_in[v].reshape(expand[i])
        axis = axis + (i,)
      r_bwd = r_bwd.at[var, r + 1].set(
          jax.nn.logsumexp(fwd + logr_mat[r], axis=axis)
      )
    f_mat = f_mat.at[r + 1].set(
        jax.nn.logsumexp(fwd_in[var] + r_bwd[var, r + 1])
    )
    r_bwd = r_bwd.at[:, r + 1].add(
        -jax.nn.logsumexp(fwd_in + r_bwd[:, r + 1], axis=1, keepdims=True)
    )
  bwd_out = bwd_fut - jax.nn.logsumexp(fwd_in + bwd_fut, axis=1, keepdims=True)

  # Combine backward messages and reward backward messages
  r_bwd = r_bwd.at[:, 0].set(bwd_out)
  f_mat = f_mat.at[0].set(f_matfut)
  if additive_reward:
    bwd_out = jax.nn.logsumexp(r_bwd + f_mat.reshape(1, -1, 1), axis=1)
    f_mat = jax.nn.logsumexp(f_mat)  # total reward
  else:
    bwd_out = r_bwd.sum(1)
    f_mat = f_mat.sum()  # total reward
  k = f_mat - jax.nn.logsumexp(fwd_in + bwd_out, axis=1).sum()
  # So that jax.nn.logsumexp(fwd_in + bwd_out, 1).sum() == f_mat
  bwd_out += k / n_e
  return f_mat, bwd_out.clip(min_val, jnp.inf)


@functools.partial(jax.jit, static_argnames=("t_cnxns"))
def update_xp(
    fwd_in: jax.Array,
    mess_jx_b_out: list[jax.Array],
    t_cnxns: CnxStructure,
    mess_xp_f_out: list[jax.Array],
) -> tuple[jax.Array, list[jax.Array], jax.Array]:
  """Update the outgoing messages from the state variables.

  Args:
    fwd_in: Forward incoming message into the "front" of the slice
    mess_jx_b_out: Backward outgoing messages from joint parent variables
    t_cnxns: Connectivity structure for the dynamics
    mess_xp_f_out: Forward outgoing messages from the state variables

  Returns:
    Tuple of
      Updated backward outgoing message out of the "front" of the slice
      Updated forward outgoing messages from the state variables
      Message delta in the update
  """
  n_e, n_s = fwd_in.shape

  f_cnxns_mat, b_cnxn_s_mat, cnxnt_mat = t_cnxns
  bwd_out = jnp.zeros((n_e, n_s))
  delta_sum = jnp.array(0.0)
  for e in range(n_e):
    # Update incoming messages
    ## Backward
    mess_xp_b_inc_e = jnp.zeros((len(b_cnxn_s_mat[e][0]), n_s))
    for i, (jx_e, idx) in enumerate(
        zip(*b_cnxn_s_mat[e])
    ):  # If this entity has any children
      mess_xp_b_inc_e = mess_xp_b_inc_e.at[i].set(mess_jx_b_out[jx_e][idx])

    # Update outgoing messages
    ## Forward
    bel = fwd_in[e] + mess_xp_b_inc_e.sum(0)
    new_mess = jnp.zeros((len(b_cnxn_s_mat[e][0]), n_s))
    for i in range(new_mess.shape[0]):
      new_mess = new_mess.at[i].set(bel - mess_xp_b_inc_e[i])
    new_mess -= new_mess.max(1, keepdims=True)
    delta_sum += jnp.abs(new_mess - mess_xp_f_out[e]).sum()
    mess_xp_f_out[e] = new_mess

    ## Backward
    bwd_out = bwd_out.at[e].set(mess_xp_b_inc_e.sum(0))
  return bwd_out, mess_xp_f_out, delta_sum


@functools.partial(jax.jit, static_argnames="t_cnxns")
def update_jx(
    mess_xp_f_out: list[jax.Array],
    mess_xa_b_inc: list[jax.Array],
    logt_mat: list[jax.Array],
    t_cnxns: CnxStructure,
    eps: float,
    mess_jx_f_out: list[jax.Array],
    mess_jx_b_out: list[jax.Array],
) -> tuple[list[jax.Array], list[jax.Array], jax.Array, jax.Array]:
  """Update the outgoing messages from the joint parent variables.

  Args:
    mess_xp_f_out: Forward outgoing messages from the state variables
    mess_xa_b_inc: Factor between the joint parent variables and the action
      after the backwards incoming messages has been marginalized
    logt_mat: Log of transition matrices for the dynamics
    t_cnxns: Connectivity structure for the dynamics
    eps: Current epsilon value (see paper)
    mess_jx_f_out: Forward outgoing messages from joint parent variables
    mess_jx_b_out: Backward outgoing messages from joint parent variables

  Returns:
    Tuple of
      Forward outgoing messages from joint parent variables
      Backward outgoing messages from joint parent variables
      "Downward" incoming messages from the action variables
      Message delta in the update
  """
  n_e, n_s, n_a = (
      len(logt_mat),
      logt_mat[0].shape[-1],
      mess_xa_b_inc[0].shape[-1],
  )

  f_cnxns_mat, b_cnxn_s_mat, cnxnt_mat = t_cnxns
  mess_xa_d_inc = jnp.zeros((n_e, n_a))
  delta_sum = jnp.array(0.0)
  for e in range(n_e):
    # Update incoming messages
    ## Forward
    f_cnxn, idx, expand, compress = f_cnxns_mat[e]
    if f_cnxn:  # If this entity has any parents
      n_v = len(f_cnxn)
      assert n_v > 0
      mess_jx_f_inc = jnp.zeros((n_s,) * n_v)
      for i, v in enumerate(f_cnxn):
        mess_jx_f_inc += mess_xp_f_out[v][idx[i]].reshape(expand[i])
    else:
      mess_jx_f_inc = jnp.zeros((n_s,) * 0)

    ## Backward
    for i in range(n_e):
      if i == e:
        continue
      mess_xa_f_inc = mess_jx_f_out[i][..., None]
      mess_xa_d_inc = mess_xa_d_inc.at[e].add(
          logsumexp_e(mess_xa_f_inc + mess_xa_b_inc[i], eps, axis=cnxnt_mat[i])
      )
    mess_jx_b_inc = logsumexp_e(
        mess_xa_d_inc[e] + mess_xa_b_inc[e], eps, axis=-1
    )

    # Update outgoing messages
    ## Forward
    new_mess = eps * mess_jx_f_inc + (eps - 1) * mess_jx_b_inc
    new_mess -= new_mess.max()
    delta = new_mess - mess_jx_f_out[e]
    delta_sum += jnp.abs(delta).sum()
    mess_jx_f_out[e] = new_mess

    ## Backward
    bel = mess_jx_b_inc + mess_jx_f_inc
    new_mess = jnp.zeros((len(f_cnxn), n_s))
    for i, v in enumerate(f_cnxn):
      new_mess = new_mess.at[i].set(
          logsumexp_e(bel, 1.0, axis=compress[i]) - mess_xp_f_out[v][idx[i]]
      )
    new_mess -= new_mess.max(1, keepdims=True)
    delta = new_mess - mess_jx_b_out[e]
    delta_sum += jnp.abs(delta).sum()
    mess_jx_b_out[e] = new_mess

  return mess_jx_f_out, mess_jx_b_out, mess_xa_d_inc, delta_sum


@functools.partial(jax.jit, static_argnames="from_qxa")
def update_xa(
    mess_jx_f_out: list[jax.Array],
    mess_xa_b_inc: list[jax.Array],
    mess_xa_d_inc: jax.Array,
    logt_mat: list[jax.Array],
    eps: float,
    min_val: float,
    from_qxa: bool,
) -> tuple[jax.Array, jax.Array, list[jax.Array]]:
  """Update the outgoing messages from the xa variables.

  Args:
    mess_jx_f_out: Forward outgoing messages from joint parent variables
    mess_xa_b_inc: Factor between the joint parent variables and the action
      after the backwards incoming messages has been marginalized
    mess_xa_d_inc: "Downward" incoming messages from the action variables
    logt_mat: Log of transition matrices for the dynamics
    eps: Current epsilon value (see paper)
    min_val: Minimum value that messages can have (log-space)
    from_qxa: Whether to derive the message updates through qxa (numerically
      different)

  Returns:
    Tuple of
      Forward outgoing message out of the "back" of the slice
      Expectation of backward messages
      log of the pseudomarginals at xa factors
  """
  n_e, n_s = len(logt_mat), logt_mat[0].shape[-1]

  fwd_out = jnp.zeros((n_e, n_s))
  e_mat_logq_mat_xa = jnp.zeros(n_e)
  log_qxa = []
  for e in range(n_e):

    # Update incoming messages
    mess_xa_f_inc_e = mess_jx_f_out[e][..., None]
    log_qxa_ = logsoftmax_e(
        mess_xa_f_inc_e + mess_xa_b_inc[e] + mess_xa_d_inc[e], eps, axis=None
    )
    e_mat_logq_mat_xa = e_mat_logq_mat_xa.at[e].set(
        (jnp.exp(log_qxa_) * mess_xa_b_inc[e]).sum()
    )
    log_qxa.append(log_qxa_)

    # Update outgoing messages
    ## Forward
    if from_qxa:
      new_fwd_out = jax.nn.logsumexp(
          (log_qxa_ - mess_xa_b_inc[e]).reshape(-1, 1)
          + logt_mat[e].reshape(-1, n_s),
          axis=0,
      )
    else:
      new_fwd_out = (1 / eps) * (mess_xa_d_inc[e] + mess_xa_f_inc_e) + (
          1 / eps - 1
      ) * mess_xa_b_inc[e]
      new_fwd_out = jax.nn.logsumexp(
          new_fwd_out.reshape(-1, 1) + logt_mat[e].reshape(-1, n_s), axis=0
      )  # alt 1

    new_fwd_out = jax.nn.log_softmax(new_fwd_out).clip(min_val, 0)
    fwd_out = fwd_out.at[e].set(new_fwd_out)

  return fwd_out, e_mat_logq_mat_xa, log_qxa


@functools.partial(jax.jit, static_argnames="cvx")
def score1step(
    fwd_in: jax.Array,
    bwd_out: jax.Array,
    cvx: bool,
    e_mat_logq_mat_xa: jax.Array,
    log_qxa: list[jax.Array],
) -> jax.Array:
  """Score the value (combined reward from all future steps) for a single slice.

  Args:
    fwd_in: Forward incoming message into the "front" of the slice
    bwd_out: Backward outgoing message out of the "front" of the slice
    cvx:  Whether to use the weighting numbers from the convex approximation
    e_mat_logq_mat_xa: Expectation of backward messages from "update_xa"
    log_qxa: log of the pseudomarginals at xa factors

  Returns:
    Estimated, untransformed value (combined reward from all future steps)
  """
  n_e = fwd_in.shape[0]
  ## Score
  f_matfut = jnp.array(0.0)
  for e in range(n_e):
    log_qxa_ = log_qxa[e]
    fwd_in_e, bwd_out_e = fwd_in[e], bwd_out[e]
    f_matfut += e_mat_logq_mat_xa[e]
    log_q = jax.nn.log_softmax(fwd_in_e + bwd_out_e)
    f_matfut += (jnp.exp(log_q) * (fwd_in_e - log_q)).sum()
    n_xv = len(log_qxa_.shape) - 1
    if n_xv > 0 and not cvx:
      log_qx = jax.nn.logsumexp(log_qxa_, axis=-1)
      f_matfut += -(jnp.exp(log_qx) * log_qx).sum()
      for i in range(n_xv):
        axis = tuple(j for j in range(n_xv) if j != i)
        log_qxv = jax.nn.logsumexp(log_qx, axis=axis)
        f_matfut += (jnp.exp(log_qxv) * log_qxv).sum()
  return f_matfut


@functools.partial(jax.jit, static_argnames=("t_cnxns", "cvx", "from_qxa"))
def vbp1step(
    fwd_in: jax.Array,
    bwd_in: jax.Array,
    logt_mat: list[jax.Array],
    t_cnxns: CnxStructure,
    cvx: bool,
    eps: float,
    min_val: float,
    from_qxa: bool,
    max_inner_iter: int,
    tol: float,
) -> tuple[float, list[jax.Array], jax.Array, jax.Array, float, int]:
  """Perform VBP within a single time slice.

  Args:
    fwd_in: Forward incoming message into the "front" of the slice
    bwd_in: Backward incoming message into the "back" of the slice
    logt_mat: Log of transition matrices for the dynamics
    t_cnxns: Connectivity structure for the dynamics
    cvx: Whether to use the weighting numbers from the convex approximation
    eps: Current epsilon value (see paper)
    min_val: Minimum value that messages can have (log-space)
    from_qxa: Whether to derive the message updates through qxa (numerically
      different)
    max_inner_iter: Maximum number of iterations to solve a time slice
    tol: Tolerance for the detection of VBP having converged in the slice

  Returns:
    Tuple of
      Estimated, untransformed value (combined reward from all future steps)
      log of the pseudomarginals at xa factors
      Forward outgoing message out of the "back" of the slice
      Backward outgoing message out of the "front" of the slice
      Message delta in the last iteration
      Number of iterations that were run
  """
  (n_e, n_s), n_a = fwd_in.shape, logt_mat[0].shape[-2]
  mess_xa_b_inc = [
      jax.nn.logsumexp(logt_mat[e] + bwd_in[e], axis=-1) for e in range(n_e)
  ]
  f_cnxns_mat, b_cnxn_s_mat, cnxnt_mat = t_cnxns
  mess_xp_f_out = [
      jnp.zeros((len(b_cnxn_s_mat[e][0]), n_s)) for e in range(n_e)
  ]
  mess_jx_f_out = [
      jnp.zeros((n_s,) * len(f_cnxns_mat[e][0])) for e in range(n_e)
  ]
  mess_jx_b_out = [jnp.zeros((len(f_cnxns_mat[e][0]), n_s)) for e in range(n_e)]
  mess = mess_xp_f_out, mess_jx_f_out, mess_jx_b_out

  def step(carry):
    mess, _, iter_ = carry
    mess_xp_f_out, mess_jx_f_out, mess_jx_b_out = mess
    delta = 0.0
    bwd_out, mess_xp_f_out, delta_ = update_xp(
        fwd_in, mess_jx_b_out, t_cnxns, mess_xp_f_out
    )
    delta += delta_
    mess_jx_f_out, mess_jx_b_out, mess_xa_d_inc, delta_ = update_jx(
        mess_xp_f_out,
        mess_xa_b_inc,
        logt_mat,
        t_cnxns,
        eps,
        mess_jx_f_out,
        mess_jx_b_out,
    )
    delta += delta_
    mess = mess_xp_f_out, mess_jx_f_out, mess_jx_b_out
    return mess, delta, iter_ + 1

  mess, delta, iter_ = jax.lax.while_loop(
      lambda carry: (carry[2] < max_inner_iter) & ((carry[1] > tol)),
      step,
      (mess, jnp.inf, 0),
  )

  mess_xp_f_out, mess_jx_f_out, mess_jx_b_out = mess
  bwd_out, mess_xp_f_out, delta_ = update_xp(
      fwd_in, mess_jx_b_out, t_cnxns, mess_xp_f_out
  )

  mess_jx_f_out, mess_jx_b_out, mess_xa_d_inc, delta_ = update_jx(
      mess_xp_f_out,
      mess_xa_b_inc,
      logt_mat,
      t_cnxns,
      eps,
      mess_jx_f_out,
      mess_jx_b_out,
  )

  fwd_out, e_mat_logq_mat_xa, log_qxa = update_xa(
      mess_jx_f_out,
      mess_xa_b_inc,
      mess_xa_d_inc,
      logt_mat,
      eps,
      min_val,
      from_qxa,
  )

  ## Backward
  bwd_out -= bwd_out.max(1, keepdims=True)
  bwd_out = bwd_out.clip(min_val, 0.0)

  ## Score
  f_matfut = score1step(fwd_in, bwd_out, cvx, e_mat_logq_mat_xa, log_qxa)

  return f_matfut, log_qxa, fwd_out, bwd_out, delta, iter_ + 1
