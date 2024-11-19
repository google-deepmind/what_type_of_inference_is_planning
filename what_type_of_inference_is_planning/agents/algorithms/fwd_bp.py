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

"""Foward belief propagation algorithm."""
# pylint: disable=g-explicit-length-test  (for better readability)
# pylint: disable=unused-variable  (for better readability)
import functools
import jax
import jax.numpy as jnp
import numpy as np

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


@functools.partial(jax.jit, static_argnames=("r_cnxns"))
def compute_reward(
    fwd_in: jax.Array, logr_mat: list[jax.Array], r_cnxns: CnxStructure
) -> float:
  """Computes total expected reward (approximated).

  Args:
    fwd_in: Forward incoming messages into the time slice
    logr_mat: Log of the analogous to the transition matrices for the rewards
    r_cnxns: Conectivity structure for the rewards

  Returns:
    Total expected reward
  """
  n_e, n_s = fwd_in.shape
  n_r = len(logr_mat)
  assert len(r_cnxns) == 3
  assert (len(logr_mat) == 0) or (logr_mat[0].shape[0] == n_s)

  ## Handling rewards
  f_mat = jnp.zeros(n_r)  # rewards current timestep
  f_cnxns_mat = r_cnxns[0]
  for r in range(n_r):
    r_bwd = jnp.zeros((n_e, n_s))
    f_cnxn, idx, expand, compress = f_cnxns_mat[r]
    assert f_cnxn  # Reward must have parents
    n_v = len(f_cnxn)
    fwd = jnp.zeros((n_s,) * n_v)
    for i, v in enumerate(f_cnxn):
      fwd += fwd_in[v].reshape(expand[i])
    f_mat = f_mat.at[r].set(jax.nn.logsumexp(fwd + logr_mat[r], axis=None))
  # Combine backward messages and reward backward messages
  f_mat = jax.nn.logsumexp(f_mat)  # total reward
  return f_mat


@functools.partial(jax.jit, static_argnames=("t_cnxns"))
def update_xp(
    fwd_in: jax.Array, t_cnxns: CnxStructure, mess_xp_f_out: list[jax.Array]
) -> list[jax.Array]:
  """Updates forward outgoing messages from the state variables.

  Args:
    fwd_in: Forward incoming messages into the time slice
    t_cnxns: Conectivity structure for the transitions
    mess_xp_f_out: Forward outgoing messages from the state variables

  Returns:
    Updated forward outgoing messages from the state variables
  """
  n_e, n_s = fwd_in.shape
  f_cnxns_mat, b_cnxns_mat, cnxnt_mat = t_cnxns
  for e in range(n_e):
    # Update outgoing messages
    ## Forward
    bel = fwd_in[e]
    new_mess = jnp.zeros((len(b_cnxns_mat[e][0]), n_s))
    for i in range(new_mess.shape[0]):
      new_mess = new_mess.at[i].set(bel)
    mess_xp_f_out[e] = new_mess
  return mess_xp_f_out


@functools.partial(jax.jit, static_argnames=("t_cnxns"))
def update_jx(
    mess_xp_f_out: jax.Array,
    logt_mat: list[jax.Array],
    t_cnxns: CnxStructure,
    mess_jx_f_out: list[jax.Array],
) -> list[jax.Array]:
  """Updates fowrward out messages from joint parent variables.

  Args:
    mess_xp_f_out: Forward outgoing messages from the state variables
    logt_mat: Log of transition matrices for the dynamics
    t_cnxns: Conectivity structure for the transitions
    mess_jx_f_out: Forward outgoing messages from joint parent variables

  Returns:
    Updated forward outgoing messages from joint parent variables
  """
  n_e, n_s = len(logt_mat), logt_mat[0].shape[-1]

  f_cnxns_mat, b_cnxns_mat, cnxnt_mat = t_cnxns
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

    # Update outgoing messages
    ## Forward
    new_mess = mess_jx_f_inc
    mess_jx_f_out[e] = new_mess

  return mess_jx_f_out


@jax.jit
def update_xa(
    mess_jx_f_out: jax.Array, actions_ev: jax.Array, logt_mat: list[jax.Array]
) -> jax.Array:
  """Updates forward out messages.

  Args:
    mess_jx_f_out: Forward outgoing messages from joint parent variables
    actions_ev: Evidence (log-space) for each of the actions at all timesteps
    logt_mat: Log of transition matrices for the dynamics

  Returns:
    Forward messages outgoing the whole time slice
  """
  n_e, n_s = len(logt_mat), logt_mat[0].shape[-1]

  fwd_out = jnp.zeros((n_e, n_s))
  for e in range(n_e):
    # Update incoming messages
    mess_xa_f_inc_e = mess_jx_f_out[e][..., None]
    # Update outgoing messages
    new_fwd_out = actions_ev + mess_xa_f_inc_e
    new_fwd_out = jax.nn.logsumexp(
        new_fwd_out.reshape(-1, 1) + logt_mat[e].reshape(-1, n_s), axis=0
    )
    new_fwd_out = jax.nn.log_softmax(new_fwd_out)
    fwd_out = fwd_out.at[e].set(new_fwd_out)
  return fwd_out


@functools.partial(jax.jit, static_argnames="t_cnxns")
def fwdbp1step(
    fwd_in: jax.Array,
    logt_mat: list[jax.Array],
    t_cnxns: CnxStructure,
    actions_ev: jax.Array,
) -> jax.Array:
  """Computes a single step of forward BP.

  Args:
    fwd_in: Incoming forward message
    logt_mat: Log of transition matrices for the dynamics
    t_cnxns: Conectivity structure for the transitions
    actions_ev: Evidence (log-space) for each of the actions at all timesteps

  Returns:
    Outgoing forward message
  """
  (n_e, n_s), n_a = fwd_in.shape, logt_mat[0].shape[-2]
  f_cnxns_mat, b_cnxns_mat, cnxnt_mat = t_cnxns
  mess_xp_f_out = [jnp.zeros((len(b_cnxns_mat[e][0]), n_s)) for e in range(n_e)]
  mess_jx_f_out = [
      jnp.zeros((n_s,) * len(f_cnxns_mat[e][0])) for e in range(n_e)
  ]

  mess_xp_f_out = update_xp(fwd_in, t_cnxns, mess_xp_f_out)
  mess_jx_f_out = update_jx(mess_xp_f_out, logt_mat, t_cnxns, mess_jx_f_out)
  fwd_out = update_xa(mess_jx_f_out, actions_ev, logt_mat)
  return fwd_out


@functools.partial(jax.jit, static_argnames=("t_cnxns", "r_cnxns", "n_steps"))
def fwdbpallsteps(
    mess_fwd: jax.Array,
    logt_mat: list[jax.Array],
    t_cnxns: CnxStructure,
    logr_mat: list[jax.Array],
    r_cnxns: CnxStructure,
    actions_ev: jax.Array,
    n_steps: int,
) -> tuple[jax.Array, jax.Array]:
  """Compiled BP forward pass over all timesteps.

  Args:
    mess_fwd: Initial forward message at the first time step.
    logt_mat: Log of transition matrices for the dynamics
    t_cnxns: Conectivity structure for the transitions
    logr_mat: Log of the analogous to the transition matrices for the rewards
    r_cnxns: Conectivity structure for the rewards
    actions_ev: Evidence (log-space) for each of the actions at all timesteps
    n_steps: Number of steps to unroll the MDP for

  Returns:
    (forward messages at all timesteps, expected reward at all timesteps)
  """
  @jax.jit
  def step(mess_fwd, t):
    new_mess_fwd = fwdbp1step(mess_fwd, logt_mat, t_cnxns, actions_ev[t])
    f_mat = compute_reward(new_mess_fwd, logr_mat, r_cnxns)
    return new_mess_fwd, f_mat

  mess_fwd, f_mat = jax.lax.scan(step, mess_fwd, jnp.arange(n_steps - 1))
  return mess_fwd, f_mat


def fwdbp(
    start: np.ndarray,
    transf: list[np.ndarray],
    transf_dep: list[np.ndarray],
    reward: list[np.ndarray],
    reward_dep: list[np.ndarray],
    n_steps: int,
    actions: np.ndarray | int | None = None,
    reward_allt: bool = True,
) -> tuple[jax.Array, jnp.ndarray]:
  """Run a single forward pass of BP, with no evidence.

    This is enough to make BP converge. Then compute the approximate expected
    reward based on the marginals. This is known as ARollout (Cui et al. 2015).

  Args:
    start: Initial state of each variable in the factored MDP
    transf: Transition matrices for the dynamics
    transf_dep: transf_dep[i] lists the variables that variable i depends
      on in the DBN dynamics.
    reward: Analogous to the transition matrices for the rewards
    reward_dep: reward_dep[r] lists the variables that reward r depends on
    n_steps: Number of steps to unroll the factored MDP for
    actions: action or list of actions to force in the factored MDP
    reward_allt: score reward at all time steps instead of just the last one

  Returns:
    (approximate expected reward, all forward messages)

  """

  n_e = len(transf)
  n_r = len(reward)
  n_a, n_s = transf[0].shape[-2:]
  assert start.shape == (n_e,)
  assert len(transf_dep) == n_e
  assert len(reward_dep) == n_r

  min_reward = 1.0
  min_reward = min([reward[r].min() for r in range(n_r)]) - min_reward

  t_cnxns = connections(transf_dep, n_e)
  r_cnxns = connections(reward_dep, n_e)
  logt_mat = [jnp.log(transf[i]) for i in range(n_e)]
  logr_mat = [jnp.log(reward[i] - min_reward) for i in range(n_r)]

  actions_ev = jnp.zeros((n_steps - 1, n_a))
  if actions is not None:
    if isinstance(actions, int):
      actions = (actions,)
    actions_ev = actions_ev.at[jnp.arange(len(actions))].set(-jnp.inf)
    actions_ev = actions_ev.at[jnp.arange(len(actions)), actions].set(0)

  # Fix starting state
  mess_fwd = (
      jnp.zeros((n_e, n_s))
      .at[:]
      .set(-jnp.inf)
      .at[jnp.arange(n_e), start]
      .set(0)
  )
  f_mat0 = compute_reward(mess_fwd, logr_mat, r_cnxns)
  mess_fwd, f_mat = fwdbpallsteps(
      mess_fwd, logt_mat, t_cnxns, logr_mat, r_cnxns, actions_ev, n_steps
  )
  f_mat = jnp.hstack((f_mat0, f_mat))

  f_mat = f_mat if reward_allt else f_mat[-1:]
  acc_r = jnp.exp(f_mat).sum()
  acc_r += min_reward * n_r * len(f_mat)

  return acc_r, mess_fwd
