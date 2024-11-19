# what_type_of_inference_is_planning
This repository contains the code used for evaluating different inference
approaches for planning on the 2011 International Probabilistic Planning
Competition in the paper "What type of inference is planning?".

Paper link: https://www.arxiv.org/abs/2406.17863<br />
Poster link: https://nips.cc/virtual/2024/poster/95030


Although code that parallelizes the evaluation is not included in this release,
it is recommended to implement a wrapper that runs agents on different instances
and seeds in parallel on multiple machines.

## Installation
Create virtual environment and install packages

```
# git clone repository
git clone https://github.com/google-deepmind/what_type_of_inference_is_planning.git
cd what_type_of_inference_is_planning

# create virtual environment
python3 -m venv venv
source venv/bin/activate

# install packages
pip install -r requirements.txt
pip install -e .
```

On a GPU or TPU machine, install jax for corresponding hardware. See https://jax.readthedocs.io/en/latest/installation.html.


Clone AISPFS repository to third_party and convert spudd files to npz files.
The [AISPFS github repository](https://github.com/Zhennan-Wu/AISPFS) includes the
spudd file for the 2011 International Probabilistic Planning Competition (IPPC)
and some of its utility functions are used for parsing spudd files.

AISPFS paper link: https://proceedings.mlr.press/v186/wu22a

```
mkdir third_party
cd third_party
git clone https://github.com/Zhennan-Wu/AISPFS.git
unzip AISPFS/spudd_sperseus.zip -d ../benchmarks/

# Converting to npz may take around 20 minutes.
cd ../benchmarks
mkdir npz
python3 create_npz.py
```


## Usage
Examples of evaluating agents on task instances. See paper for more details.

```
# change directory to what_type_of_inference_is_planning/what_type_of_inference_is_planning
cd ../what_type_of_inference_is_planning

# See all flag options by running python3 evaluate_agent.py --help

# Agent that picks random action.
python3 evaluate_agent.py --task_name=traffic_inst_mdp --inst_id=1 --seed=0 --agent=random
# Agent that uses variation inference with LP solver. (VI LP in paper)
python3 evaluate_agent.py --task_name=game_of_life_inst_mdp --inst_id=1 --seed=0 --agent=vilp
# Agent that uses forward belief propagation. Equivelent to ARollout in this
# setting. (ARollout in paper)
python3 evaluate_agent.py --task_name=sysadmin_inst_mdp --inst_id=1 --seed=0 --agent=fwdbp
# Agent that uses value belief propagation. (VBP in paper)
python3 evaluate_agent.py --task_name=skill_teaching_inst_mdp --inst_id=1 --seed=0 --agent=vbp
```

The cumulative rewards are stored in separate files under the result folder for
each run. The file name is:

`/results/ipc_results_{horizon}_{agent}/{taks_name}__{inst_id}_{seed}.txt`

The following is the file structure of the folder
what_type_of_inference_is_planning.

```
├── agents
│   ├── agent_base.py
│   ├── agent_fwdbp.py
│   ├── agent_random.py
│   ├── agent_vbp.py
│   ├── agent_vilp.py
│   └── algorithms
│       ├── bin_planner.py
│       ├── fwd_bp.py
│       ├── updates.py
│       └── value_bp.py
├── envs
│   └── environment.py
└── evaluate_agent.py
```

- The file envs/environment.py contains code that models the environment.
- The agents folder contains the four different agents used in the experiment.
- The agents/algorithms folder contains inference code used by the agents.

## Citing this work

```latex
@inproceedings{
lazaro2024what,
title={What type of inference is planning?},
author={L{\'a}zaro-Gredilla, Miguel and Ku, Li Yang and Murphy, Kevin P and George, Dileep},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=TXsRGrzICz}
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
