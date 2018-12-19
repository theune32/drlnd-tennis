from unityagents import UnityEnvironment
import numpy as np
import torch

from collections import deque
from agent import Agent
from tbWrapper import TBWrapper
from datetime import datetime

env = UnityEnvironment(file_name='files/Tennis.app')
brain_name = env.brain_names[0]
env_info = env.reset(train_mode=True)[brain_name]
brain = env.brains[brain_name]
agent_A = Agent(state_size=env_info.vector_observations.shape[1], action_size=brain.vector_action_space_size, random_seed=10, agent_count=1)
agent_B = Agent(state_size=env_info.vector_observations.shape[1], action_size=brain.vector_action_space_size, random_seed=10, agent_count=1)
tag = "test"
tensorboard = TBWrapper('./logs/logs-{}-{}'.format(tag, datetime.now))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


def ddpg(n_episodes=500):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agent_A.reset()
        agent_B.reset()
        score = np.zeros(num_agents)
        while True:
            action_A = agent_A.act(state[0].reshape((1, 24)))
            action_B = agent_B.act(state[1].reshape((1, 24)))
            assert action_A.shape == (1, 2)
            env_info = env.step(np.vstack([action_A, action_B]))[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            agent_A.step(state[0:], action_A, reward, next_state[0:], done)
            agent_B.step(state[1:], action_A, reward, next_state[1:], done)
            state = next_state
            score += np.array(reward)
            tensorboard.add_scalar("{}-actor".format(tag), agent_A.loss[0])
            tensorboard.add_scalar("{}-critic".format(tag), agent_B.loss[1])
            if np.any(done):
                break
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {}\tScore: {}'.format(i_episode, np.mean(scores_deque), np.mean(score)),
              end="")
        if i_episode % 100 == 0 or np.mean(scores_deque) > 30:
            torch.save(agent_A.actor_local.state_dict(), 'checkpoint_actor-{}.pth'.format(i_episode))
            torch.save(agent_B.actor_local.state_dict(), 'checkpoint_actor-{}.pth'.format(i_episode))
            torch.save(agent_A.critic_local.state_dict(), 'checkpoint_critic-{}.pth'.format(i_episode))
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) > 30:
            torch.save(agent_A.actor_local.state_dict(), 'checkpoint_actor_30+-{}.pth'.format(i_episode))
            torch.save(agent_B.actor_local.state_dict(), 'checkpoint_actor_30+-{}.pth'.format(i_episode))
            torch.save(agent_A.critic_local.state_dict(), 'checkpoint_critic_30+-{}.pth'.format(i_episode))
            np.save("scores_30+-{}".format(i_episode), scores)
            break

    return scores


scores = ddpg()

