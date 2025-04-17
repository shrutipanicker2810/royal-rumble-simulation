# import numpy as np
# import random
# import math
# import pygame
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# # Import your environment classes.
# from wrestling_battle_royale import BattleRoyaleEnv, Wrestler
#
# # === PPO Hyperparameters ===
# OBS_DIM = 7  # [health, stamina, last_action, self_pos(x,y), opponent_pos(x,y)]
# ACT_DIM = 4  # Actions: [Punch, Kick, No-op, Signature]
# LR = 3e-4
# GAMMA = 0.99
# EPSILON = 0.2
# C1 = 0.5  # Value loss coefficient
# C2 = 0.01  # Entropy bonus coefficient
# UPDATE_EPOCHS = 4
# BATCH_SIZE = 32
#
#
# # === Define the PPO Network ===
# class PPONetwork(nn.Module):
#     def __init__(self, obs_dim, act_dim):
#         super(PPONetwork, self).__init__()
#         self.fc1 = nn.Linear(obs_dim, 64)
#         self.fc2 = nn.Linear(64, 64)
#         # Actor head: outputs logits for each action
#         self.actor = nn.Linear(64, act_dim)
#         # Critic head: outputs a state value
#         self.critic = nn.Linear(64, 1)
#
#     def forward(self, x):
#         x = torch.tanh(self.fc1(x))
#         x = torch.tanh(self.fc2(x))
#         logits = self.actor(x)
#         value = self.critic(x)
#         return logits, value
#
#
# # === Define PPO Agent ===
# class PPOAgent:
#     def __init__(self, obs_dim, act_dim, lr=LR):
#         self.model = PPONetwork(obs_dim, act_dim)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
#
#     def select_action(self, obs):
#         """
#         Given an observation (numpy array of shape [OBS_DIM]), choose an action.
#         Returns a tuple: (action, log_prob, value).
#         """
#         obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # Shape: [1, obs_dim]
#         logits, value = self.model(obs_tensor)
#         probs = torch.softmax(logits, dim=-1)
#         dist = torch.distributions.Categorical(probs)
#         action = dist.sample()
#         log_prob = dist.log_prob(action)
#         return action.item(), log_prob, value.item()
#
#     def update(self, trajectories, update_epochs=UPDATE_EPOCHS, batch_size=BATCH_SIZE,
#                gamma=GAMMA, epsilon=EPSILON):
#         """
#         Update the PPO policy using collected trajectory transitions.
#         Each trajectory is a dictionary with keys:
#           'obs', 'action', 'log_prob', 'reward', 'done', 'value', 'return', 'advantage'
#         """
#         obs = torch.FloatTensor([t['obs'] for t in trajectories])
#         actions = torch.LongTensor([t['action'] for t in trajectories])
#         old_log_probs = torch.stack([t['log_prob'] for t in trajectories]).detach()
#         returns = torch.FloatTensor([t['return'] for t in trajectories])
#         advantages = torch.FloatTensor([t['advantage'] for t in trajectories])
#
#         for _ in range(update_epochs):
#             # Shuffle and split the trajectories into mini-batches
#             indices = np.arange(len(trajectories))
#             np.random.shuffle(indices)
#             for i in range(0, len(trajectories), batch_size):
#                 batch_idx = indices[i:i + batch_size]
#                 batch_obs = obs[batch_idx]
#                 batch_actions = actions[batch_idx]
#                 batch_old_log_probs = old_log_probs[batch_idx]
#                 batch_returns = returns[batch_idx]
#                 batch_advantages = advantages[batch_idx]
#
#                 logits, values = self.model(batch_obs)
#                 values = values.squeeze()
#                 probs = torch.softmax(logits, dim=-1)
#                 dist = torch.distributions.Categorical(probs)
#                 new_log_probs = dist.log_prob(batch_actions)
#                 entropy = dist.entropy().mean()
#
#                 ratio = torch.exp(new_log_probs - batch_old_log_probs)
#                 surr1 = ratio * batch_advantages
#                 surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * batch_advantages
#                 actor_loss = -torch.min(surr1, surr2).mean()
#                 critic_loss = nn.MSELoss()(values, batch_returns)
#                 loss = actor_loss + C1 * critic_loss - C2 * entropy
#
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#         return loss.item()
#
#
# # === Helper function to compute returns and advantages ===
# def compute_returns(trajectories, gamma=GAMMA):
#     """
#     Given a list of transitions (which may come from different wrestlers),
#     group transitions by wrestler and compute the discounted return (and advantage)
#     for each transition.
#     """
#     # Group trajectories by wrestler ID
#     grouped = {}
#     for t in trajectories:
#         wid = t['wrestler_id']
#         grouped.setdefault(wid, []).append(t)
#
#     # For each wrestler’s trajectory, compute discounted returns backward in time.
#     for traj in grouped.values():
#         R = 0
#         # Process in reverse order for each individual wrestler
#         for t in reversed(traj):
#             if t['done']:
#                 R = 0
#             R = t['reward'] + gamma * R
#             t['return'] = R
#             t['advantage'] = t['return'] - t['value']
#
#     # Flatten the grouped transitions
#     all_transitions = []
#     for traj in grouped.values():
#         all_transitions.extend(traj)
#     return all_transitions
#
#
# # === PPO Training Loop ===
# def train_ppo(num_episodes=100, max_steps_per_episode=200):
#     # Create the PPO agent
#     ppo_agent = PPOAgent(OBS_DIM, ACT_DIM, lr=LR)
#
#     # Initialize the Battle Royale environment.
#     # Note: We use the same environment initialization as in your wrestling_battle_royale.py.
#     env = BattleRoyaleEnv(ring_size=3.0, entry_interval=5)
#     wrestlers_data = [
#         ("Roman Reigns", 10, 191, 120, 88),
#         ("Brock Lesnar", 9, 191, 130, 95),
#         ("Seth Rollins", 9, 185, 98, 80),
#         ("Becky Lynch", 9, 168, 61, 77),
#         ("Finn Bálor", 7, 180, 86, 65),
#         ("Kevin Owens", 7, 183, 122, 70),
#         ("AJ Styles", 7, 180, 99, 73),
#         ("Dominik Mysterio", 5, 185, 91, 49),
#         ("Liv Morgan", 5, 160, 50, 37),
#         ("Otis", 4, 178, 150, 18)
#     ]
#     env.wrestlers = [Wrestler(env, name, i, pop, height, weight, exp)
#                      for i, (name, pop, height, weight, exp) in enumerate(wrestlers_data)]
#     random.shuffle(env.wrestlers)
#     # Set normalization bounds from wrestlers
#     heights = [w.height for w in env.wrestlers]
#     weights = [w.weight for w in env.wrestlers]
#     Wrestler.height_min = min(heights)
#     Wrestler.height_max = max(heights)
#     Wrestler.weight_min = min(weights)
#     Wrestler.weight_max = max(weights)
#     env.reset()
#     # Start with two wrestlers in the match
#     env._add_new_wrestler()
#     env._add_new_wrestler()
#
#     episode_rewards = []
#
#     for episode in range(num_episodes):
#         trajectories = []  # To hold rollout data across all wrestlers (shared policy)
#         episode_reward = 0
#         step = 0
#
#         # Run an episode until termination (e.g. when a winner is determined)
#         while True:
#             step += 1
#             actions = {}
#             current_obs = env._get_obs()  # Dictionary: key = wrestler id, value = observation (size 7)
#             if not current_obs:
#                 break  # No active wrestlers
#             step_transitions = []
#
#             # For each active wrestler, select an action using the PPO model.
#             for wid, ob in current_obs.items():
#                 action, log_prob, value = ppo_agent.select_action(ob)
#                 actions[wid] = action
#                 step_transitions.append({
#                     'wrestler_id': wid,
#                     'obs': ob,
#                     'action': action,
#                     'log_prob': log_prob,
#                     'value': value,
#                     'reward': 0,  # to be filled after step
#                     'done': False  # to be filled after step
#                 })
#
#             # Take a simulation step in the environment.
#             new_obs, rewards, dones, infos, initiator, responder = env.step(actions)
#
#             # Record rewards and done flags for each active wrestler.
#             for trans in step_transitions:
#                 wid = trans['wrestler_id']
#                 trans['reward'] = rewards.get(wid, 0)
#                 trans['done'] = dones.get(wid, False)
#                 episode_reward += trans['reward']
#                 trajectories.append(trans)
#
#             # Terminate the episode if any wrestler signals done, or if max steps reached.
#             if any(dones.values()) or step >= max_steps_per_episode or len(env.active_wrestlers) <= 1:
#                 break
#
#         # Compute returns and advantages for the collected transitions.
#         trajectories = compute_returns(trajectories, gamma=GAMMA)
#
#         # Update the PPO agent with the collected experience.
#         loss = ppo_agent.update(trajectories, update_epochs=UPDATE_EPOCHS, batch_size=BATCH_SIZE,
#                                 gamma=GAMMA, epsilon=EPSILON)
#         episode_rewards.append(episode_reward)
#
#         print(
#             f"Episode {episode + 1}/{num_episodes} - Total Reward: {episode_reward:.2f}, Loss: {loss:.4f}, Steps: {step}")
#
#         # Reset the environment for the next episode and add two new wrestlers.
#         env.reset()
#         env._add_new_wrestler()
#         env._add_new_wrestler()
#
#     # Save the trained model parameters.
#     torch.save(ppo_agent.model.state_dict(), "ppo_wrestling_model.pth")
#     print("Training completed and model saved as ppo_wrestling_model.pth")
#
#
# if __name__ == "__main__":
#     pygame.init()  # Initialize pygame (used within the environment)
#     train_ppo(num_episodes=100)  # Adjust the number of episodes as needed
#
# import numpy as np
#
#
# def get_action_probability_matrix(action_matrix):
#     """
#     Given an action_matrix of shape (10,10,3) where each entry accumulates rewards
#     from interactions between wrestlers (initiator vs. responder) for each attack type,
#     compute the corresponding probability matrix using softmax along the actions axis.
#
#     Args:
#         action_matrix (np.ndarray): A matrix of shape (10,10,3) with reward values.
#
#     Returns:
#         np.ndarray: A probability matrix of shape (10,10,3) where for each pair (i, j),
#                     the probabilities over the 3 actions sum to 1.
#     """
#     # Initialize a probability matrix with the same shape.
#     prob_matrix = np.zeros_like(action_matrix, dtype=float)
#
#     # Loop over each initiator-responder pair.
#     for i in range(action_matrix.shape[0]):
#         for j in range(action_matrix.shape[1]):
#             # Retrieve the reward vector for the (i, j) pair.
#             rewards = action_matrix[i, j, :]
#             # For numerical stability, subtract the maximum reward.
#             max_reward = np.max(rewards)
#             exp_rewards = np.exp(rewards - max_reward)
#             sum_exp = np.sum(exp_rewards)
#             # If sum_exp is zero (or extremely small), default to uniform distribution.
#             if sum_exp < 1e-8:
#                 prob_matrix[i, j, :] = np.ones(3) / 3.0
#             else:
#                 prob_matrix[i, j, :] = exp_rewards / sum_exp
#     return prob_matrix
#
#
# # Example usage:
# if __name__ == "__main__":
#     # Assume we have an action_matrix from your environment (10x10x3)
#     action_matrix = np.random.randn(10, 10, 3)  # Example: random values; replace with your matrix.
#     prob_matrix = get_action_probability_matrix(action_matrix)
#     print("Action Probability Matrix:")
#     print(prob_matrix)


import numpy as np
import random
import math
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# Import your environment classes.
from wrestling_battle_royale import BattleRoyaleEnv, Wrestler

# === PPO Hyperparameters ===
OBS_DIM = 4    # [init_health, init_stamina, resp_health, resp_stamina]
ACT_DIM = 3    # Actions: [Punch, Kick, Signature]
LR = 3e-4
GAMMA = 0.99
EPSILON = 0.2
C1 = 0.5       # Value loss coefficient
C2 = 0.01      # Entropy bonus coefficient
UPDATE_EPOCHS = 4
BATCH_SIZE = 32

# === 1) Wrap the env to return our 4‑dim state ===
class SimplifiedBattleRoyaleEnv(BattleRoyaleEnv):
    def _get_obs(self):
        obs = {}
        for w in self.active_wrestlers:
            ih, is_ = w.health, w.stamina
            if w._opponents:
                # find closest opponent just to choose which one to report
                opp = min(
                    w._opponents,
                    key=lambda o: np.linalg.norm(self.positions[w.match_pos] - self.positions[o.match_pos])
                )
                rh, rs = opp.health, opp.stamina
            else:
                rh, rs = 0.0, 0.0
            obs[w.id] = np.array([ih, is_, rh, rs], dtype=np.float32)
        return obs

# === 2) PPO Network with 4‑dim input, 3 actions ===
class PPONetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1    = nn.Linear(OBS_DIM, 64)
        self.fc2    = nn.Linear(64, 64)
        self.actor  = nn.Linear(64, ACT_DIM)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.actor(x), self.critic(x)

# === 3) PPO Agent ===
class PPOAgent:
    def __init__(self):
        self.model     = PPONetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def select_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)        # [1,4]
        logits, value = self.model(obs_t)                  # logits:[1,3], value:[1,1]
        probs  = torch.softmax(logits, dim=-1)             # [1,3]
        dist   = torch.distributions.Categorical(probs)
        a      = dist.sample()                             # scalar
        return a.item(), dist.log_prob(a), value.item()

    def update(self, trajectories, update_epochs=UPDATE_EPOCHS,
               batch_size=BATCH_SIZE, gamma=GAMMA, epsilon=EPSILON):
        obs          = torch.FloatTensor([t['obs']       for t in trajectories])
        actions      = torch.LongTensor( [t['action']    for t in trajectories])
        old_log_probs= torch.stack(    [t['log_prob']   for t in trajectories]).detach()
        returns      = torch.FloatTensor([t['return']    for t in trajectories])
        advantages   = torch.FloatTensor([t['advantage'] for t in trajectories])

        for _ in range(update_epochs):
            idx = np.arange(len(trajectories))
            np.random.shuffle(idx)
            for i in range(0, len(trajectories), batch_size):
                batch = idx[i:i+batch_size]
                b_obs  = obs[batch]
                b_act  = actions[batch]
                b_olp  = old_log_probs[batch]
                b_ret  = returns[batch]
                b_adv  = advantages[batch]

                logits, values = self.model(b_obs)
                values = values.squeeze()
                probs  = torch.softmax(logits, dim=-1)
                dist   = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(b_act)
                entropy= dist.entropy().mean()

                ratio   = torch.exp(new_lp - b_olp)
                s1      = ratio * b_adv
                s2      = torch.clamp(ratio, 1-epsilon, 1+epsilon) * b_adv
                a_loss  = -torch.min(s1, s2).mean()
                c_loss  = nn.MSELoss()(values, b_ret)
                loss    = a_loss + C1 * c_loss - C2 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

# === Helper to compute returns & advantages ===
def compute_returns(trajectories, gamma=GAMMA):
    grouped = {}
    for t in trajectories:
        grouped.setdefault(t['wrestler_id'], []).append(t)
    for traj in grouped.values():
        R = 0
        for t in reversed(traj):
            if t['done']:
                R = 0
            R = t['reward'] + gamma * R
            t['return']    = R
            t['advantage'] = R - t['value']
    # flatten
    return [t for traj in grouped.values() for t in traj]

# === PPO Training Loop ===
def train_ppo(num_episodes=100, max_steps_per_episode=200):
    agent = PPOAgent()
    env   = SimplifiedBattleRoyaleEnv(ring_size=3.0, entry_interval=5)

    # set up wrestlers exactly as before...
    wrestlers_data = [
        ("Roman Reigns", 10, 191, 120, 88),
        ("Brock Lesnar", 9, 191, 130, 95),
        ("Seth Rollins", 9, 185, 98, 80),
        ("Becky Lynch", 9, 168, 61, 77),
        ("Finn Bálor",   7, 180, 86, 65),
        ("Kevin Owens",  7, 183, 122,70),
        ("AJ Styles",    7, 180, 99, 73),
        ("Dominik Mysterio",5,185,91,49),
        ("Liv Morgan",   5, 160,50, 37),
        ("Otis",         4, 178,150,18)
    ]
    env.wrestlers = [Wrestler(env, n, i, p, h, w, e)
                     for i,(n,p,h,w,e) in enumerate(wrestlers_data)]
    random.shuffle(env.wrestlers)

    # normalization bounds...
    heights = [w.height for w in env.wrestlers]
    weights = [w.weight for w in env.wrestlers]
    Wrestler.height_min = min(heights)
    Wrestler.height_max = max(heights)
    Wrestler.weight_min = min(weights)
    Wrestler.weight_max = max(weights)

    # kickoff each episode
    for ep in range(num_episodes):
        obs     = []
        total_r = 0
        step    = 0
        env.reset()
        env._add_new_wrestler()
        env._add_new_wrestler()

        while True:
            step += 1
            current = env._get_obs()
            if not current:
                break

            # select actions
            batch = []
            actions = {}
            for wid, st in current.items():
                a, lp, val = agent.select_action(st)
                actions[wid] = a
                batch.append({
                    'wrestler_id': wid, 'obs': st,
                    'action': a, 'log_prob': lp,
                    'value': val, 'reward': 0.0, 'done': False
                })

            new_obs, rewards, dones, *_ = env.step(actions)
            for trans in batch:
                wid = trans['wrestler_id']
                trans['reward'] = rewards.get(wid, 0.0)
                trans['done']   = dones.get(wid, False)
                total_r += trans['reward']
                obs.append(trans)

            if any(dones.values()) or step >= max_steps_per_episode or len(env.active_wrestlers)<=1:
                break

        traj = compute_returns(obs)
        loss = agent.update(traj)
        print(f"Episode {ep+1}/{num_episodes}  Reward={total_r:.2f}  Loss={loss:.4f}")

    # save model
    torch.save(agent.model.state_dict(), "ppo_wrestling_model.pth")
    print("Training complete.")

if __name__ == "__main__":
    pygame.init()
    train_ppo(num_episodes=100)


import numpy as np

def get_action_probability_matrix(action_matrix):
    prob_matrix = np.zeros_like(action_matrix, dtype=float)
    for i in range(action_matrix.shape[0]):
        for j in range(action_matrix.shape[1]):
            rewards = action_matrix[i, j, :]
            m = np.max(rewards)
            e = np.exp(rewards - m)
            s = np.sum(e)
            prob_matrix[i, j, :] = (e / s) if s>1e-8 else np.ones(3)/3
    return prob_matrix

if __name__ == "__main__":
    # Assume we have an action_matrix from your environment (10x10x3)
    action_matrix = np.random.randn(10, 10, 3)  # Example: random values; replace with your matrix.
    prob_matrix = get_action_probability_matrix(action_matrix)
    print("Action Probability Matrix:")
    print(prob_matrix)


