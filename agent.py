import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 5e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of the actor
LR_CRITIC = 1e-4  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SharedCritic:
    """Shared critic version of the agent. Assumes 2 agents."""
    def __init__(self, state_size, action_size, random_seed, agent_count):
        """Initialize an Agent object"""

        np.random.seed(42)
        torch.manual_seed(42)

        self.agent_count = agent_count
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.loss = [0., 0., 0.]

        self.actor_local_a = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target_a = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer_a = optim.Adam(self.actor_local_a.parameters(), lr=LR_ACTOR)

        self.actor_local_b = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target_b = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer_b = optim.Adam(self.actor_local_b.parameters(), lr=LR_ACTOR)

        self.critic_local = Critic(state_size * 2, action_size * 2, random_seed, agent_count).to(device)
        self.critic_target = Critic(state_size * 2, action_size * 2, random_seed, agent_count).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.noise = OUNoise((self.agent_count, self.action_size))

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    # noinspection PyUnresolvedReferences
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local_a.eval()
        self.actor_local_b.eval()

        with torch.no_grad():
            action_a = self.actor_local_a(state[0]).cpu().data.numpy()
            action_b = self.actor_local_b(state[1]).cpu().data.numpy()
        self.actor_local_a.train()
        self.actor_local_b.train()
        actions = np.vstack([action_a, action_b])
        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # -------------- update Critic -------------- #
        actions_next_a = self.actor_target_a(next_states[:, :self.state_size])
        actions_next_b = self.actor_target_b(next_states[:, self.state_size:])
        actions_next = torch.cat([actions_next_a, actions_next_b], 1)

        Q_targets_next = self.critic_target(next_states, actions_next)

        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------------- update Actor a and b -------------- #
        actions_pred_a = self.actor_local_a(states[:, :self.state_size])
        actions_pred_b = self.actor_local_b(states[:, self.state_size:])

        # loss per actor based predicted actions of agent a and the actual actions of agent b, and vice versa
        actor_loss_a = -self.critic_local(states, torch.cat([actions_pred_a, actions[:, 2:]], 1))[:, 0].mean()
        actor_loss_b = -self.critic_local(states, torch.cat([actions[:, :2], actions_pred_b], 1))[:, 1].mean()

        self.actor_optimizer_a.zero_grad()
        self.actor_optimizer_b.zero_grad()
        actor_loss_a.backward()
        actor_loss_b.backward()
        self.actor_optimizer_a.step()
        self.actor_optimizer_b.step()

        # ---------- update target network ---------- #
        self.loss = [actor_loss_a.item(), actor_loss_b.item(), critic_loss.item()]
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local_a, self.actor_target_a, TAU)
        self.soft_update(self.actor_local_b, self.actor_target_b, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = copy.copy(self.mu)

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(size=self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Buffer to store experiences, size equal to the buffer_size"""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initializes the ReplayBuffer
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience the replay buffer"""
        e = self.experience(state.flatten(), action.flatten(), reward, next_state.flatten(), done)
        self.memory.append(e)

    # noinspection PyUnresolvedReferences
    def sample(self):
        """Random sample of experiences from memory of size equal to the batch_size (or as specified)"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of the memory"""
        return len(self.memory)
