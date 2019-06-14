# Google Football Research
![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)   
This is the pytorch example of the google football research, more contents will be updated soon:soccer:!
## Requirements
- python-3.6.8
- openai-baselines
- pytorch-1.1.0
- [gfootball](https://github.com/google-research/football)
## TODO List
- [ ] add more tasks and examples - full game is in plan.
- [ ] remove openai-baseline's functions.
- [ ] add more algorithms: **IMPALA** and **Ape-X DQN**.
- [ ] add multi-agent reinforcement learning algorithms (**MARL**)

## Installation
Please install the `gfootball` according to the instructions [here](https://github.com/google-research/football).
1. Make sure your `pip` is less than `19`, the lastest version of `pip` will disable `--process-dependency-links`.
```bash
conda install pip==18.1
```
2. Install tensorflow (well, we don't need it, but it's required for `gfootball`).
```bash
pip install tensorflow
```
3. Install gfootball.
``` bash 
git clone https://github.com/google-research/football.git
cd football
pip install .[tf_cpu] --process-dependency-links (we don't need GPU for tensorflow)
```
## How to use the code
Train the simple example - **academy_empty_goal_close**
```bash
python train_example.py --cuda (if you have a GPU)
```
Play the demo:
```bash
python demo.py
```
## Demo
academy_empty_goal_close| academy_run_to_score
-----------------------|-----------------------|
![](figures/academy_empty_goal_close.gif)| ![](figures/academy_run_to_score.gif)

academy_3_vs_1_with_keeper| academy_counterattack_easy
-----------------------|-----------------------|
![](figures/academy_3_vs_1_with_keeper.gif)| ![](figures/academy_counterattack_easy.gif)
