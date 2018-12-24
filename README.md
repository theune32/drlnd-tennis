# DRLND: Collaboration and Competition Project

In this project the goal is to learn how to let 2 agents, handling rackets, cooperate to try and keep a ball in the air.

The Unity environment has an observation space of 24 parameters (3 stacked frames of 8 vector observation parameters
for one agent and the ball) per agent. The action space has a size of 2 per agent, with values ranging from -1 to 1.
For the rewards structure is as follows: a score of +0.1 per ball that was hit by and agent, any dropped ball results
in a reward of -0.01.

The goal is to have an average score of more than +0.5 over 100 episodes.

## Setup
This repo can be cloned from [https://github.com/theune32/drlnd-tennis] and subsequently run in the root:

    make setup
    
The basic setup was shared by a colleague and does the following:
* create "files" directory
* Download the unity environment app: "Tennis.app"
* create a venv based on the "requirements.txt"
* download the "Tennis.ipynb" from the Udacity repo, can be used by:
`jupyter notebook`

To start a training session, just run `python train.py`

The repository contains 4 files:
* train.py: here the training, setup and logging are handled
* agent.py: all interactions with the model are handled here
* model.py: contains the model itself
* tbWrapper.py: contains some wrapper functionality for tensorboardX

Tensorboard can be run by
`tensorboard --logdir` and accessing the dashboard from your browser [localhost:6006]
The log files for tensorboard are stored in /logs and contain information about the loss of both Actors and the Critic
and both the score and average score (100 episodes) over time (steps).

Saved model weights when finished are stored in the following checkpoint files:
- checkpoint_actor_a_05+-{episode number}.pth
- checkpoint_actor_b_05+-{episode number}.pth
- checkpoint_critic_05+-{episode number}.pth

The complete list of scores for a successful run are stored in: scores_05+-{episode number}.npy.

End results and conclusions can be found in the "Report.md" file.