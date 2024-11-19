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

"""Evaluate an agent on a benchmark."""
from collections.abc import Sequence
import os
import time

from absl import app
from absl import flags
from absl import logging
import jax
from jax import random

from what_type_of_inference_is_planning.agents import agent_fwdbp
from what_type_of_inference_is_planning.agents import agent_random
from what_type_of_inference_is_planning.agents import agent_vbp
from what_type_of_inference_is_planning.agents import agent_vilp
from what_type_of_inference_is_planning.envs import environment


N_EXTRA_STEPS = 39

_HORIZON = flags.DEFINE_integer("horizon", default=10, help="Horizon.")

_TASK_NAME = flags.DEFINE_string(
    "task_name", default="crossing_traffic_inst_mdp", help="Task name to test."
)
_INST_ID = flags.DEFINE_integer("inst_id", default=1, help="Inst ID to test.")

_SEED = flags.DEFINE_integer("seed", default=0, help="Seed for environment.")

_AGENT = flags.DEFINE_string(
    "agent", default="random", help="Agent to test. [vilp, vbp, fwdbp, random]"
)

_USE_RAND = flags.DEFINE_bool(
    "use_rand",
    default=False,
    help=(
        "Use pre-generated random numbers for environment. Text files with"
        " random numbers for each test instance and seed need to be placed in"
        " /benchmarks/rands/ folder if set to True."
    ),
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARKS_DIR = os.path.dirname(ROOT_DIR) + "/benchmarks/npz/"
RANDS_DIR = os.path.dirname(ROOT_DIR) + "/benchmarks/rands/"
RESULTS_DIR = os.path.dirname(ROOT_DIR) + "/results/"
jax.config.update("jax_enable_x64", True)


def main(argv: Sequence[str]) -> None:
  del argv
  start_time = time.time()

  dataset = (
      BENCHMARKS_DIR
      + _TASK_NAME.value
      + "__"
      + str(_INST_ID.value)
      + ".npz"
  )
  seed = _SEED.value
  env = environment.Environment(dataset)
  if _AGENT.value == "vilp":
    agent = agent_vilp.AgentVILP(env, horizon=_HORIZON.value)
  elif _AGENT.value == "fwdbp":
    agent = agent_fwdbp.AgentFWDBP(env, horizon=_HORIZON.value)
  elif _AGENT.value == "random":
    agent = agent_random.AgentRandom(env)
  elif _AGENT.value == "vbp":
    agent = agent_vbp.AgentVBP(env, horizon=_HORIZON.value)
  else:
    raise ValueError("Unknown agent: %s" % _AGENT.value)
  logging.info("------ %s ---- seed %d ---", dataset, seed)
  rng = random.PRNGKey(seed)
  env.reset()

  rands_list_to_use = None
  if _USE_RAND.value:
    rand_file_name = (
        RANDS_DIR
        + "rand_"
        + _TASK_NAME.value
        + "__"
        + str(_INST_ID.value)
        + "_"
        + str(_SEED.value)
        + ".txt"
    )
    assert _HORIZON.value <= 40
    rands_list_to_use = []
    with open(rand_file_name, "r") as fin:
      fin.readline()

      for _ in range(39):
        float_strings = fin.readline().split(" ")[1:-1]
        rands_list_to_use.append([float(s) for s in float_strings])

  final_reward = env.run(
      rng,
      agent,
      n_extra_steps=N_EXTRA_STEPS,
      rands_list_to_use=rands_list_to_use
  )
  result = (dataset, seed, final_reward)
  logging.info("results: %s", str(result))
  logging.info("time: %f", time.time() - start_time)
  result_dir = (
      "ipc_results_"
      + str(_HORIZON.value)
      + "_"
      + str(_AGENT.value)
  )

  out_file_name = (
      RESULTS_DIR
      + result_dir
      + "/"
      + _TASK_NAME.value
      + "__"
      + str(_INST_ID.value)
      + "_"
      + str(_SEED.value)
      + ".txt"
  )

  os.makedirs(os.path.dirname(out_file_name), exist_ok=True)
  with open(out_file_name, "w") as fout:
    fout.write(str(final_reward))


if __name__ == "__main__":
  app.run(main)
