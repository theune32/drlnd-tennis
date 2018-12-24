from unityagents import UnityEnvironment
import numpy as np
import torch

from collections import deque
from agent import Agent, SharedCritic
from tbWrapper import TBWrapper
from datetime import datetime

env = UnityEnvironment(file_name='files/Tennis.app')
brain_name = env.brain_names[0]
env_info = env.reset(train_mode=True)[brain_name]
brain = env.brains[brain_name]
agent = SharedCritic(state_size=env_info.vector_observations.shape[1], action_size=brain.vector_action_space_size,
                     random_seed=10, agent_count=2)
tag = "test"
tensorboard = TBWrapper('./logs/logs-{}-{}'.format(tag, datetime.now()))
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


def ddpg(n_episodes=5000):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        agent.reset()
        score = np.zeros(2)
        while True:
            actions = agent.act(state)
            assert actions.shape == (2, 2)
            env_info = env.step(actions)[brain_name]
            next_state = env_info.vector_observations
            reward = env_info.rewards
            done = env_info.local_done
            agent.step(state, actions, reward, next_state, done)
            state = next_state
            score += np.array(reward)
            tensorboard.add_scalar("{}-actor_a".format(tag), agent.loss[0])
            tensorboard.add_scalar("{}-actor_b".format(tag), agent.loss[1])
            tensorboard.add_scalar("{}-critic".format(tag), agent.loss[2])
            if np.any(done):
                score = np.max(score)
                break
        scores_deque.append(score)
        scores.append(score)
        tensorboard.add_scalar("{}-score".format(tag), score)
        tensorboard.add_scalar("{}-100-ep-scores".format(tag), np.mean(scores_deque))
        print('\rEpisode {}\tAverage Score: {}\tScore: {}'.format(i_episode, np.mean(scores_deque), score))
        if i_episode % 100 == 0 or np.mean(scores_deque) > 0.5:
            torch.save(agent.actor_local_a.state_dict(), 'checkpoint_actor_a-{}.pth'.format(i_episode))
            torch.save(agent.actor_local_b.state_dict(), 'checkpoint_actor_b-{}.pth'.format(i_episode))
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic-{}.pth'.format(i_episode))
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) > 0.5:
            torch.save(agent.actor_local_a.state_dict(), 'checkpoint_actor_a_05+-{}.pth'.format(i_episode))
            torch.save(agent.actor_local_b.state_dict(), 'checkpoint_actor_b_05+-{}.pth'.format(i_episode))
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_05+-{}.pth'.format(i_episode))
            np.save("scores_05+-{}".format(i_episode), scores)
            break

    return scores


scores = ddpg()

