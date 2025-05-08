import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.observations[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):

    def __init__(self, 
            state_dim, 
            action_dim, 
            action_std_init):
        super(ActorCritic, self).__init__()
        
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to('cuda')
        # actor
        self.actor = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_dim),
                        nn.Tanh()
                    )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to('cuda')

    def act(self, state):

        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):

        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to('cuda')
        dist = MultivariateNormal(action_mean, cov_mat)
        
        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self, total_update=1.0):
        
        action_dim = 1
        action_std_init = 0.6
        state_dim = 5
        checkpoint = "/home/chyang/workspace/DRL-Assignment-4/Q2/output/2025.05.08-15.02.38/checkpoints/ckpt_1240000"
        gamma = 0.98
        lambda_gae = 0.8
        entropy_coeff = 0.0
        eps_clip = 0.2
        lr_actor = 0.001
        lr_critic = 0.001
        train_epochs = 160
        random_mode = False
        clip_grad_norm = 0.5

        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float64)

        self.action_std = action_std_init

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.lambda_gae = lambda_gae    
        self.entropy_coeff = entropy_coeff
        self.eps_clip = eps_clip
        self.train_epochs = train_epochs
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.clip_grad_norm = clip_grad_norm
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to('cuda')
        
        self.checkpoint = checkpoint
        if self.checkpoint:
            self.policy.load_state_dict(torch.load(checkpoint))

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        total_update *= self.train_epochs
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: 1.0 - min(step, total_update) / total_update
        )

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to('cuda')
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.loss_fn = nn.MSELoss()
        
        self.random_mode = random_mode

    def act(self, observation):
        if self.random_mode:
            return self.action_space.sample()
        else:
            with torch.no_grad():
                observation = torch.FloatTensor(observation).to('cuda')
                action, action_logprob, state_val = self.policy_old.act(observation)
            if not self.checkpoint:
                self.buffer.observations.append(observation)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        
    def update(self):
        # Convert buffer contents to tensors
        rewards = self.buffer.rewards
        is_terminals = self.buffer.is_terminals
        old_obs = torch.squeeze(torch.stack(self.buffer.observations, dim=0)).detach().to('cuda')
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to('cuda')
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to('cuda')
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to('cuda')

        advantages = []
        gae = 0
        returns = []

        next_value = 0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(is_terminals[t])
            delta = rewards[t] + self.gamma * next_value * mask - old_state_values[t].item()
            gae = delta + self.gamma * self.lambda_gae * mask * gae
            advantages.insert(0, gae)
            next_value = old_state_values[t].item()
            returns.insert(0, gae + old_state_values[t].item())

        advantages = torch.tensor(advantages, dtype=torch.float32).to('cuda')
        returns = torch.tensor(returns, dtype=torch.float32).to('cuda')
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        for _ in range(self.train_epochs):
            # Evaluate old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_obs, old_actions)
            state_values = torch.squeeze(state_values)

            # Calculate ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Total loss
            loss = -torch.min(surr1, surr2) \
                + 0.5 * self.loss_fn(state_values, returns) \
                - self.entropy_coeff * dist_entropy

            # Update
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            self.lr_scheduler.step()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()