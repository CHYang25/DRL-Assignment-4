from eval import make_env
from student_agent import Agent
import wandb
import click
import torch
import random
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm

@click.command()
@click.option('--seed', default=42, help='Random seed for reproducibility.')
@click.option('--max-steps', default=1_250_000, help='Total environment steps to train for.')
@click.option('--update-timestep', default=4000, help='Number of steps between each policy update.')
@click.option('--action-std-decay-freq', default=250000, help='Frequency of decaying action std.')
@click.option('--action-std-decay-rate', default=0.05, help='Rate at which to decay action std.')
@click.option('--min-action-std', default=0.1, help='Minimum value for action std.')
@click.option('--log-freq', default=2000, help='Logging frequency (in steps).')
@click.option('--save-model-freq', default=20000, help='Model checkpoint saving frequency (in steps).')
def train(seed, max_steps, update_timestep, action_std_decay_freq, action_std_decay_rate, min_action_std,
            log_freq, save_model_freq):
    
    env = make_env()    

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    run_name = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")

    ckpt_dir = f"./output/{run_name}/checkpoints/"
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb_run = wandb.init(
        dir=f'./output/{run_name}',
        project='drl_hw4',
        mode="online",
        name=run_name,
    )

    ppo_agent = Agent(max_steps / update_timestep)
    steps = 0
    episode = 0

    pbar = tqdm(total=max_steps, desc="Training", unit="step")

    wandb.config.update({
        # Training Configs
        "output_dir": f'./output/{run_name}',
        "seed": seed,
        "max_steps": max_steps,
        "update_timestep": update_timestep,
        "action_std_decay_freq": action_std_decay_freq,
        "action_std_decay_rate": action_std_decay_rate,
        "min_action_std": min_action_std,
        "log_freq": log_freq,
        "save_model_freq": save_model_freq,

        # Agent and MDP configs
        "action_std": ppo_agent.action_std,
        "action_dim": ppo_agent.action_dim,
        "state_dim": ppo_agent.state_dim,
        "gamma": ppo_agent.gamma,
        "eps_clip": ppo_agent.eps_clip,
        "train_epochs": ppo_agent.train_epochs,
        "lr_actor": ppo_agent.lr_actor,
        "lr_critic": ppo_agent.lr_critic,
    })


    while steps <= max_steps:

        obs, _ = env.reset(seed=np.random.randint(0, 1000000))
        episode_step = 0
        total_reward = 0

        done = False

        while not done:
            
            action = ppo_agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            total_reward += reward
            steps +=1
            episode_step += 1
            pbar.update(1)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            # update PPO agent
            if steps % update_timestep == 0:
                ppo_agent.update()

            if steps % action_std_decay_freq == 0:
                ppo_agent.action_std = max(min_action_std, ppo_agent.action_std - action_std_decay_rate)
                ppo_agent.policy.set_action_std(ppo_agent.action_std)
                ppo_agent.policy_old.set_action_std(ppo_agent.action_std)

            if steps % log_freq == 0:
                wandb_run.log({
                    "steps": steps,
                    "action_std": ppo_agent.action_std,
                }, step=steps)

            if steps % save_model_freq == 0:
                torch.save(ppo_agent.policy_old.state_dict(), os.path.join(ckpt_dir, f"ckpt_{steps}"))

        wandb_run.log({
            'total_reward': total_reward,
            'episode': episode,
            'episode_length': episode_step
        }, step=steps)

        pbar.set_postfix({
            'episode': episode,
            'reward': total_reward,
            'steps': steps,
        })

        episode += 1
    
    pbar.close()

if __name__ == '__main__':
    train()
