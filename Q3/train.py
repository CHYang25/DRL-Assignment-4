from student_agent import Agent
import wandb
import click
import torch
import random
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm, trange
import sys
from tensordict import TensorDict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

def evaluate(agent, env, episodes=32):
    total_reward = 0.0
    for _ in trange(episodes, desc=f"Rollout {episodes}", leave=True):
        obs, _ = env.reset(seed=np.random.randint(0, 1000000))
        done = False
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
    return total_reward / episodes


@click.command()
@click.option('--seed', default=42, help='Random seed for reproducibility.')
@click.option('--max-steps', default=2_000_000, help='Total environment steps to train for.')
@click.option('--start-steps', default=10000, help='Number of initial random steps for exploration.')
@click.option('--batch-size', default=256, help='Batch size for updates.')
@click.option('--update-after', default=10000, help='Number of steps to collect transitions before starting updates.')
@click.option('--train-epochs', default=1, help="Number of training steps for each update cycle.")
@click.option('--update-every', default=1, help='Number of environment steps between updates.')
@click.option('--eval-interval', default=10000, help='Evaluation interval.')
@click.option('--save-model-freq', default=10000, help='Model checkpoint saving frequency (in steps).')
@click.option('--lr-schedule', default=False, help='Learning rate scheduler applied.')
def train(seed, max_steps, start_steps, batch_size, update_after, train_epochs, update_every, eval_interval, save_model_freq, lr_schedule):
    
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

    env = make_dmc_env("humanoid-walk", np.random.randint(0, 1000000), flatten=True, use_pixels=False)

    sac_agent = Agent(total_train_steps=((max_steps - update_after // update_every) * train_epochs) if lr_schedule else None)
    steps = 0
    episode = 0

    pbar = tqdm(total=max_steps, desc="Training", unit="step")

    wandb.config.update({
        # Training Configs
        "output_dir": f'./output/{run_name}',
        "seed": seed,
        "max_steps": max_steps,
        "start_steps": start_steps, 
        "batch_size": batch_size, 
        "update_after": update_after, 
        "update_every": update_every, 
        "eval_interval": eval_interval,
        "save_model_freq": save_model_freq,

        # Agent and MDP configs
        "action_dim": sac_agent.action_dim,
        "state_dim": sac_agent.state_dim,
        "gamma": sac_agent.gamma,
        "tau": sac_agent.tau,
        "alpha": sac_agent.alpha,
        "lr": sac_agent.lr,
    })

    while steps <= max_steps:
        
        obs, _ = env.reset(seed=np.random.randint(0, 1000000))
        episode_step = 0
        total_reward = 0

        done = False

        while not done:
            
            action = sac_agent.act(obs, random=steps < start_steps, deterministic=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            total_reward += reward
            steps +=1
            episode_step += 1
            pbar.update(1)

            # saving reward and is_terminals
            sac_agent.buffer.add(TensorDict({
                "obs": torch.from_numpy(obs).to(sac_agent.device).to(torch.float32).unsqueeze(0),
                "action": torch.from_numpy(action).to(sac_agent.device).to(torch.float32).unsqueeze(0),
                "reward": torch.tensor(reward, device=sac_agent.device, dtype=torch.float32).unsqueeze(0),
                "next_obs": torch.from_numpy(next_obs).to(sac_agent.device).to(torch.float32).unsqueeze(0),
                "done": torch.tensor(done, device=sac_agent.device, dtype=torch.float32).unsqueeze(0),
            }, batch_size=[1]))
            
            obs = next_obs.copy()

            if steps >= update_after and steps % update_every == 0:
                for _ in range(train_epochs):
                    update_log = sac_agent.update(batch_size)
                wandb_run.log(update_log, step=steps)

            if steps % eval_interval == 0:
                eval_reward = evaluate(sac_agent, env)
                wandb_run.log({
                    "eval_reward": eval_reward,
                    "steps": steps,
                }, step=steps)

            if steps % save_model_freq == 0:
                torch.save(sac_agent.policy.state_dict(), os.path.join(ckpt_dir, f"ckpt_{steps}"))
    
        wandb_run.log({
            'total_reward': total_reward,
            'episode': episode,
            'episode_length': episode_step,
        }, step=steps)

        pbar.set_postfix({
            'episode': episode,
            'reward': total_reward,
            'steps': steps,
        })

        episode += 1
    
    torch.save(sac_agent.policy.state_dict(), os.path.join(ckpt_dir, f"ckpt_final"))
    pbar.close()

if __name__ == '__main__':
    train()
