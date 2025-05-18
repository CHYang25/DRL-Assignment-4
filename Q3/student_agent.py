import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import RandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, init_weight=1E-2):
        super().__init__()
        self.linear1 = nn.Linear(state_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)
        self.action_dim = action_dim

        self.fc = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
            nn.ReLU()
        )

        # Weight initialization
        self.linear1.weight.data.uniform_(-init_weight, init_weight)
        self.linear1.bias.data.uniform_(0, init_weight)
        self.linear2.weight.data.uniform_(-init_weight, init_weight)
        self.linear2.bias.data.uniform_(0, init_weight)
        self.mean_layer.weight.data.uniform_(-init_weight, init_weight)
        self.mean_layer.bias.data.uniform_(0, init_weight)
        self.log_std_layer.weight.data.uniform_(-init_weight, init_weight)
        self.log_std_layer.bias.data.uniform_(0, init_weight)

    def forward(self, state):
        x = self.fc(state)
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)
        std = torch.ones_like(mean) * log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z).sum(1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(1, keepdim=True)
        return action, log_prob

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, init_weight=1E-2):
        super().__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        self.q_net = nn.Sequential(
            self.linear1,
            nn.ReLU(),
            self.linear2,
            nn.ReLU(),
            self.linear3
        )

        self.linear1.weight.data.uniform_(-init_weight, init_weight)
        self.linear1.bias.data.uniform_(0, init_weight)
        self.linear2.weight.data.uniform_(-init_weight, init_weight)
        self.linear2.bias.data.uniform_(0, init_weight)
        self.linear3.weight.data.uniform_(-init_weight, init_weight)
        self.linear3.bias.data.uniform_(0, init_weight)


    def forward(self, state, action):
        return self.q_net(torch.cat([state, action], dim=-1))


# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self, total_train_steps=None):
        
        action_dim = 21
        state_dim = 67
        checkpoint = "./output/2025.05.18-14.00.19/checkpoints/ckpt_1390000"
        gamma = 0.99
        tau = 5e-3
        lr = 3e-4
        reward_scale = 20
        random_mode = False
        device = 'cpu' 

        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.reward_scale = reward_scale
        
        self.buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(1_000_000, device=device),
            sampler=RandomSampler(),
        )

        self.policy = PolicyNetwork(state_dim, action_dim).to(device)

        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.q1_target = QNetwork(state_dim, action_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.log_alpha = torch.log(torch.ones(1, device=device)).requires_grad_(True)  # Learnable log_alpha
        self.alpha = torch.exp(self.log_alpha.detach())
        self.entropy_target = -1. * torch.tensor(action_dim, device=device, dtype=torch.float)

        self.checkpoint = checkpoint
        if self.checkpoint:
            self.policy.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.total_train_steps = total_train_steps
        if total_train_steps:
            self.policy_scheduler = CosineAnnealingLR(self.policy_optimizer, T_max=self.total_train_steps)
            self.q1_scheduler = CosineAnnealingLR(self.q1_optimizer, T_max=self.total_train_steps)
            self.q2_scheduler = CosineAnnealingLR(self.q2_optimizer, T_max=self.total_train_steps)
        
        self.loss_fn = nn.MSELoss()
        self.random_mode = random_mode
        self.device = device

    def act(self, observation, random=False, deterministic=False):
        if self.random_mode or random:
            return self.action_space.sample()
        else:
            observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            with torch.no_grad():
                if deterministic:
                    mean, _ = self.policy(observation)
                    action = torch.tanh(mean).cpu().detach().numpy()[0]
                else:
                    action, _ = self.policy.sample(observation)
                    action = action.cpu().detach().numpy()[0]
                
            # action += np.random.normal(0, 0.2, self.action_dim)
            # action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
                
            return action
        
    def update(self, batch_size=256):

        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = batch['obs'].squeeze(1), batch['action'].squeeze(1), batch['reward'], batch['next_obs'].squeeze(1), batch['done']
        
        sampled_actions, logprobs = self.policy.sample(states)

        alpha_loss = -(self.log_alpha * (logprobs + self.entropy_target).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = torch.exp(self.log_alpha.detach())
        
        with torch.no_grad():
            next_sampled_actions, next_logprobs = self.policy.sample(next_states)
            target_q1 = self.q1_target(next_states, next_sampled_actions)
            target_q2 = self.q2_target(next_states, next_sampled_actions)
            target_q, _ = torch.min(torch.cat((target_q1, target_q2), dim=-1), dim=-1, keepdim=True) 
            target_q -= self.alpha * next_logprobs
            target_value = rewards + (1 - dones) * self.gamma * target_q

        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)
        q1_loss = self.loss_fn(current_q1, target_value)
        q2_loss = self.loss_fn(current_q2, target_value)
        q_loss = 0.5 * (q1_loss + q2_loss)

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        
        q1_pi = self.q1(states, sampled_actions)
        q2_pi = self.q2(states, sampled_actions)
        min_q_pi, _ = torch.min(torch.cat((q1_pi, q2_pi), dim=-1), dim=-1, keepdim=True)
        policy_loss = (self.alpha * logprobs - min_q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if self.total_train_steps:
            self.policy_scheduler.step()
            self.q1_scheduler.step()
            self.q2_scheduler.step()

        # Soft update of target networks
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)

        return {
            "q1_loss": q1_loss.detach().cpu().item(),
            "q2_loss": q2_loss.detach().cpu().item(),
            "policy_loss": policy_loss.detach().cpu().item(),
            "alpha_loss": alpha_loss.detach().cpu().item(),
            "avg_q1": current_q1.mean().detach().cpu().item(),
            "alpha": self.alpha.detach().cpu().item(),
            "lr_policy": self.policy_scheduler.get_last_lr()[0],
            "lr_q1": self.q1_scheduler.get_last_lr()[0],
            "lr_q2": self.q2_scheduler.get_last_lr()[0],
        }
