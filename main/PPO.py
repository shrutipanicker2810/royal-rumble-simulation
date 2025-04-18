import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from wrestling_battle_royale import BattleRoyaleEnv, Wrestler


OBS_DIM       = 6     # [init_id, init_health, init_stamina, resp_id, resp_health, resp_stamina]
ACT_DIM       = 3     # Actions: [Punch, Kick, Signature]
LR            = 3e-4
GAMMA         = 0.99
EPSILON       = 0.2
C1            = 0.5   # value loss coeff
C2            = 0.01  # entropy bonus coeff
UPDATE_EPOCHS = 4
BATCH_SIZE    = 32


class PPONetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1    = nn.Linear(obs_dim, 64)
        self.fc2    = nn.Linear(64, 64)
        self.actor  = nn.Linear(64, act_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.actor(x), self.critic(x)

# PPO Agent
class PPOAgent:
    def __init__(self):
        self.model     = PPONetwork(OBS_DIM, ACT_DIM)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def select_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)   # [1, OBS_DIM]
        logits, value = self.model(obs_t)             # logits:[1,ACT_DIM], value:[1,1]
        probs  = torch.softmax(logits, dim=-1)        # [1,ACT_DIM]
        dist   = torch.distributions.Categorical(probs)
        a      = dist.sample()                       # scalar
        return a.item(), dist.log_prob(a), value.item()

    def update(self, trajectories):
        obs          = torch.FloatTensor([t['obs']       for t in trajectories])
        actions      = torch.LongTensor( [t['action']    for t in trajectories])
        old_log_probs= torch.stack(    [t['log_prob']   for t in trajectories]).detach()
        returns      = torch.FloatTensor([t['return']    for t in trajectories])
        advantages   = torch.FloatTensor([t['advantage'] for t in trajectories])

        for _ in range(UPDATE_EPOCHS):
            idx = np.arange(len(trajectories))
            np.random.shuffle(idx)
            for start in range(0, len(idx), BATCH_SIZE):
                batch = idx[start:start+BATCH_SIZE]
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
                entropy = dist.entropy().mean()

                ratio   = torch.exp(new_lp - b_olp)
                s1      = ratio * b_adv
                s2      = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * b_adv
                a_loss  = -torch.min(s1, s2).mean()
                c_loss  = nn.MSELoss()(values, b_ret)
                loss    = a_loss + C1 * c_loss - C2 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

# Compute discounted returns and advantages
def compute_returns(transitions):
    grouped = {}
    for t in transitions:
        grouped.setdefault(t['wrestler_id'], []).append(t)
    for traj in grouped.values():
        R = 0
        for t in reversed(traj):
            if t['done']:
                R = 0
            R = t['reward'] + GAMMA * R
            t['return']    = R
            t['advantage'] = R - t['value']
    return [t for traj in grouped.values() for t in traj]


def train_ppo(num_episodes=100, max_steps_per_episode=200):
    pygame.init()
    agent = PPOAgent()
    env   = BattleRoyaleEnv(ring_size=3.0, entry_interval=5)


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


    heights = [w.height for w in env.wrestlers]
    weights = [w.weight for w in env.wrestlers]
    Wrestler.height_min = min(heights)
    Wrestler.height_max = max(heights)
    Wrestler.weight_min = min(weights)
    Wrestler.weight_max = max(weights)


    for ep in range(num_episodes):
        transitions = []
        total_reward = 0.0
        step = 0

        env.reset()
        env._add_new_wrestler()
        env._add_new_wrestler()

        while True:
            step += 1

            # 1) Pick initiator & responder via SA
            initiator, responder = env._select_combatants()
            if not initiator or not responder:

                env.step({})
                if step >= max_steps_per_episode:
                    break
                else:
                    continue

            # 2) Build our 6 dim state
            obs = np.array([
                initiator.id,
                initiator.health,
                initiator.stamina,
                responder.id,
                responder.health,
                responder.stamina
            ], dtype=np.float32)

            # 3) Get action/logp/value
            action, logp, value = agent.select_action(obs)

            # 4) Step environment
            _, rewards, dones, _, init2, resp2 = env.step({initiator.id: action})

            # 5) Record transition for PPO
            r    = rewards.get(initiator.id, 0.0)
            done = dones.get(initiator.id, False)
            transitions.append({
                'wrestler_id': initiator.id,
                'obs':         obs,
                'action':      action,
                'log_prob':    logp,
                'value':       value,
                'reward':      r,
                'done':        done
            })
            total_reward += r

            # 6) Check termination
            if done or step >= max_steps_per_episode or len(env.active_wrestlers) <= 1:
                break

        # 7) Update PPO
        traj = compute_returns(transitions)
        loss = agent.update(traj)

        print(f"Episode {ep+1}/{num_episodes}  TotalReward={total_reward:.2f}  Loss={loss:.4f}")

    # Save the trained policy
    torch.save(agent.model.state_dict(), "ppo_wrestling_model.pth")
    print("Training complete. Model saved to ppo_wrestling_model.pth")


    N = len(env.wrestlers)
    matrix = np.zeros((N, N, ACT_DIM), dtype=float)
    for i, wi in enumerate(env.wrestlers):
        for j, wj in enumerate(env.wrestlers):
            if i == j:
                continue
            state = np.array([
                wi.id,
                wi.health,
                wi.stamina,
                wj.id,
                wj.health,
                wj.stamina
            ], dtype=np.float32)
            with torch.no_grad():
                logits, _ = agent.model(torch.FloatTensor(state).unsqueeze(0))
                probs     = torch.softmax(logits, dim=-1).numpy().squeeze()
            matrix[i, j, :] = probs

    print("\nAction Probability Matrix (i→j):")
    print(matrix)

if __name__ == "__main__":
    train_ppo(num_episodes=10000, max_steps_per_episode=200)
