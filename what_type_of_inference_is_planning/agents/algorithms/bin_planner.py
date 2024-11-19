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

"""VI LP planner with binary variables."""
# pylint: disable=g-explicit-length-test  (for better readability)
# pylint: disable=unused-variable  (for better readability)
import time
from typing import Sequence

from absl import flags
from absl import logging
import cvxpy as cp
import numpy as np


_SOLVER_NAME = flags.DEFINE_string(
    "solver_name",
    default="GLOP",
    help="Solver name. SCIPY or GLOP for vilp agent.",
)

VERBOSE = False


def solve_bin(
    t_mat: list[np.ndarray],
    r_mat: list[np.ndarray],
    wiring: tuple[
        list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]
    ],
    init: list[list[str | None]],
    n_steps: int,
    enums: Sequence[np.ndarray],
    enums_less1: Sequence[np.ndarray],
    laterals: tuple[int, int, np.ndarray, np.ndarray] | None = None,
    action: int | None = None,
    boolean: bool = False,
) -> tuple[float, list[list[np.ndarray]], list[list[np.ndarray]], np.ndarray]:
  """Compute the optimal solution of the VI LP problem.

  Args:
    t_mat: Transition matrices for the dynamics
    r_mat: Analogous to the transition matrices for the rewards
    wiring: Pre-multiplying matrices (forward and backward for dynamics,
      forward and backward for rewards)
    init: initial state in categorical format as needed by each target
      variable, first for dynamics, then for rewards.
    n_steps: Number of steps to unroll the factored MDP for
    enums: Sequence of variable lists. Only one variable in each list can
      be 1 at a time. I.e., XOR lists.
    enums_less1: Sequence of variable lists. At most one variable in each
    list can be 1 at a time. I.e., one of them or none of them.
    laterals: Variables across which to enforce consistent pseudomarginals
      within the same timestep, including their premultiplying matrices.
    action: First action to force in the factored MDP
    boolean: MIP. Force results to only admit boolean solutions. This solves
      the problem exactly, but it's too slow in non-trivial problems.

  Returns:
    Tuple with:
      Optimal value of the VI LP problem
      pseudomarginals over the transition factors
      pseudomarginals over the reward factors
      pseudomarginals over the actions
  """

  n_ent = len(t_mat)
  n_rew = len(r_mat)
  n_a = t_mat[0].shape[1]

  f_t_mat, b_t_mat, f_r_mat, b_r_mat = wiring
  init_t_mat, init_r_mat = init

  t_mat2 = [t_mat[i].reshape(-1, t_mat[i].shape[2]) for i in range(n_ent)]

  assert len(init_t_mat) == n_ent
  assert len(init_r_mat) == n_rew
  for i in range(n_ent):
    if b_t_mat[i].size > 0:
      assert b_t_mat[i].shape == (f_t_mat[i].shape[0], t_mat[i].shape[0])
  for i in range(n_rew):
    assert b_r_mat[i].shape == (f_r_mat[i].shape[0], r_mat[i].shape[0])
  qt_mat = [
      [cp.Variable(t_mat[i].shape[:2], nonneg=True) for i in range(n_ent)]
      for t in range(n_steps - 1)
  ]

  qr_mat = [
      [cp.Variable(r_mat[i].shape, nonneg=True) for i in range(n_rew)]
      for t in range(n_steps - 1)
  ]

  score = 0.0
  constraints = []
  if boolean:
    constraints.extend([
        cp.sum(qt_mat[t][0], axis=0) == cp.Variable(n_a, boolean=True)
        for t in range(n_steps - 1)
    ])
  if action is not None:
    constraints.append(cp.sum(qt_mat[0][0], axis=0)[action] == 1)
  for t in range(n_steps - 1):
    # Constraints
    # valid transition posterior
    constraints.extend([cp.sum(qt_mat[t][i]) == 1 for i in range(n_ent)])
    # valid reward posterior
    constraints.extend([cp.sum(qr_mat[t][i]) == 1 for i in range(n_rew)])

    # Matching across time and initial conditions
    for i in range(n_ent):
      if t == 0:
        constraints.append(cp.sum(qt_mat[t][i], axis=1)[init_t_mat[i]] == 1)
      elif b_t_mat[i].size > 0:
        qx = cp.sum(qt_mat[t][i], axis=1)
        for j in range(len(f_t_mat[i])):
          size = qt_mat[t - 1][f_t_mat[i][j]].size
          qxp = (
              cp.reshape(qt_mat[t - 1][f_t_mat[i][j]], size, "C")
              @ t_mat2[f_t_mat[i][j]]
          )
          constraints.append(qxp[1] == b_t_mat[i][j] @ qx)

    # Matching reward
    for i in range(n_rew):
      if b_r_mat[i].size > 0:
        for j in range(len(f_r_mat[i])):
          size = qt_mat[t][f_r_mat[i][j]].size
          qxp = (
              cp.reshape(qt_mat[t][f_r_mat[i][j]], size, "C")
              @ t_mat2[f_r_mat[i][j]]
          )
          constraints.append(qxp[1] == b_r_mat[i][j] @ qr_mat[t][i])

    # Matching actions
    qa = cp.sum(qt_mat[t][0], axis=0)
    for i in range(1, n_ent):
      constraints.append(qa == cp.sum(qt_mat[t][i], axis=0))

    # Adding reward
    for i in range(n_rew):
      if t == 0:
        pass
      score += cp.scalar_product(qr_mat[t][i], r_mat[i])

    # Enums
    for j in range(len(enums_less1)):
      c = 0.0
      for i in enums_less1[j]:
        size = qt_mat[t][i].size
        qxp = cp.reshape(qt_mat[t][i], size, "C") @ t_mat2[i]
        c += qxp[1]
      constraints.append(c <= 1)

    for j in range(len(enums)):
      c = 0.0
      for i in enums[j]:
        size = qt_mat[t][i].size
        qxp = cp.reshape(qt_mat[t][i], size, "C") @ t_mat2[i]
        c += qxp[1]
      constraints.append(c == 1)

    # Laterals
    if laterals is not None:
      for i, j, m_mat1, m_mat2 in laterals:
        size_1 = qt_mat[t][i].size
        size_2 = qt_mat[t][j].size
        constraints.append(
            m_mat1 @ cp.reshape(qt_mat[t][i], size_1, "C")
            == m_mat2 @ cp.reshape(qt_mat[t][j], size_2, "C")
        )

  prob = cp.Problem(cp.Maximize(score), constraints)
  start_time = time.time()

  f_mat = prob.solve(verbose=VERBOSE, solver=_SOLVER_NAME.value)
  logging.info("Solve took %.2f seconds", time.time() - start_time)

  assert prob.status == cp.OPTIMAL

  qa = np.array(
      [cp.sum(qt_mat[t][0], axis=0).value for t in range(n_steps - 1)]
  )
  qt_mat = [
      [qt_mat[t][i].value for i in range(n_ent)] for t in range(n_steps - 1)
  ]
  qr_mat = [
      [qr_mat[t][i].value for i in range(n_rew)] for t in range(n_steps - 1)
  ]

  return f_mat, qt_mat, qr_mat, qa


def get_mat_time(
    dependencies: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
  """Compute the matrices needed to enforce temporal consistency.

  Args:
    dependencies: dependencies[i] lists the variables that variable i depends
      on in the DBN

  Returns:
    forward pre-multiplying matrices and backard pre-multiplying matrices
  """
  f_elements = []
  b_matrices = []
  for state_dep in dependencies:
    shape = 2 * np.ones(len(state_dep), int)
    if len(shape) < 1:
      f_elemen = np.array([[]])
      b_matrix = np.array([[]])
    else:
      f_elemen = np.array(state_dep)
      n = shape.prod()
      indices = np.array(np.unravel_index(np.arange(n), shape))
      b_matrix = np.zeros((len(shape), n), int)
      for i in range(len(shape)):
        b_matrix[i] = indices[i] == 1
    f_elements.append(f_elemen)
    b_matrices.append(b_matrix)

  return f_elements, b_matrices


def get_init(
    dependencies: list[np.ndarray], init_state: np.ndarray
) -> list[str | None]:
  """Compute the initial states in categorical format, per target variable.

  Variables are assumed to be binary.

  Args:
    dependencies: dependencies[i] lists the variables that variable i depends
      on in the DBN
    init_state: Initial state of each variable in the factored MDP

  Returns:
    Initial state in categorical format as needed by each target variable
  """
  init = []
  for i in range(len(dependencies)):
    if dependencies[i].size > 0:
      init.append(
          np.ravel_multi_index(
              init_state[dependencies[i]], (2,) * dependencies[i].shape[0]
          )
      )
    else:
      init.append(None)
  return init
