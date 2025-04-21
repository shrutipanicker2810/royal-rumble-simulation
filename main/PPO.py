import pygame
import numpy as np
import pygame
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

from genetic_evolution import evolve_wrestlers
from wrestling_battle_royale import BattleRoyaleEnv, Wrestler


OBS_DIM       = 6     # [init_id, init_health, init_stamina, resp_id, resp_health, resp_stamina]
ACT_DIM       = 3     # Actions: [Punch, Kick, Signature]
LR            = 3e-4
gamma         = 0.99
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


class PPOAgent:
    def __init__(self):
        self.model     = PPONetwork(OBS_DIM, ACT_DIM)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def select_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        logits, value = self.model(obs_t)
        probs  = torch.softmax(logits, dim=-1)
        dist   = torch.distributions.Categorical(probs)
        a      = dist.sample()
        return a.item(), dist.log_prob(a), value.item()

    def update(self, transitions):
        obs          = torch.FloatTensor([t['obs']       for t in transitions])
        actions      = torch.LongTensor( [t['action']    for t in transitions])
        old_log_probs= torch.stack(    [t['log_prob']   for t in transitions]).detach()
        returns      = torch.FloatTensor([t['return']    for t in transitions])
        advantages   = torch.FloatTensor([t['advantage'] for t in transitions])

        for _ in range(UPDATE_EPOCHS):
            idx = np.arange(len(transitions))
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

# Compute returns & advantages
def compute_returns(transitions):
    grouped = {}
    for t in transitions:
        grouped.setdefault(t['wrestler_id'], []).append(t)
    for traj in grouped.values():
        R = 0
        for t in reversed(traj):
            if t['done']:
                R = 0
            R = t['reward'] + gamma * R
            t['return']    = R
            t['advantage'] = R - t['value']
    return [t for traj in grouped.values() for t in traj]



def train_ppo(num_episodes=100, max_steps=200):
    pygame.init()
    agent = PPOAgent()
    env   = BattleRoyaleEnv(ring_size=3.0, entry_interval=5)

    # initialize wrestlers
    data = [
        ("Roman R",10,191,120,88),("Brock L",9,191,130,95),("Seth R",9,185,98,80),
        ("Becky L",9,168,61,77),("Finn B",7,180,86,65),("Kevin O",7,183,122,70),
        ("AJ Styles",7,180,99,73),("Dominik M",5,185,91,49),("Liv M",5,160,50,37),("Otis",4,178,150,18)
    ]
    env.wrestlers = [Wrestler(env,n,i,p,h,w,e) for i,(n,p,h,w,e) in enumerate(data)]
    random.shuffle(env.wrestlers)


    heights = [w.height for w in env.wrestlers]
    weights = [w.weight for w in env.wrestlers]
    Wrestler.height_min = min(heights); Wrestler.height_max = max(heights)
    Wrestler.weight_min = min(weights); Wrestler.weight_max = max(weights)

    num_generations = 3
    for ep in range(num_episodes):
        generation = 0
        while generation < num_generations - 1:
            trans     = [] 
            total_r = 0 # In gen, calculating the reward for only last generation with evolved wrestlers
            step    = 0
            print(f"\n=== Generation {generation + 1} ===")
            env.reset()
            env._add_new_wrestler()
            env._add_new_wrestler()

            while True:
                step+=1
                init, resp = env._select_combatants()

                # 1) Pick initiator & responder via SA
                if not init or not resp:
                    env.step({});
                    # if step>=max_steps: break
                    # continue

                # 2) Build our 6 dim state
                obs = np.array([init.id,init.health,init.stamina,resp.id,resp.health,resp.stamina],dtype=np.float32)

                # 3) Get action/logp/value
                action, logp, val = agent.select_action(obs)

                # 4) Step environment
                _, rewards, dones, _,_,_ = env.step({init.id:action})

                # 5) Check termination
                d = dones.get(init.id,False)
                if d or step>=max_steps or len(env.active_wrestlers)<=1: break

            # 6) Record transition for PPO for the last generation only
            r = rewards.get(init.id,0.0); 
            trans.append({'wrestler_id':init.id,'obs':obs,'action':action,'log_prob':logp,'value':val,'reward':r,'done':d})
            total_r += r

            # 7) Evolve wrestlers if not last generation
            env.wrestlers = evolve_wrestlers(env, env.wrestlers, num_generations=1)
            generation += 1

        # 8) Update PPO
        traj = compute_returns(trans)
        loss = agent.update(traj)
        print(f"Ep {ep+1}/{num_episodes} R={total_r:.2f} L={loss:.4f}")

    torch.save(agent.model.state_dict(),"ppo_wrestling_model10000.pth")
    print("Training done.")


    N = len(env.wrestlers)
    matrix = np.zeros((N,N,ACT_DIM),dtype=float)
    names  = [w.name for w in env.wrestlers]
    for i,wi in enumerate(env.wrestlers):
        for j,wj in enumerate(env.wrestlers):
            if i==j: continue
            s = np.array([wi.id,wi.health,wi.stamina,wj.id,wj.health,wj.stamina],dtype=np.float32)
            with torch.no_grad(): logits,_ = agent.model(torch.FloatTensor(s).unsqueeze(0)); probs=torch.softmax(logits,dim=-1).numpy().squeeze()
            matrix[i,j,:]=probs

    # print labeled probability tables
    for i,name in enumerate(names):
        df = pd.DataFrame(matrix[i,:,:], index=names, columns=["Punch","Kick","Signature"])
        print(f"\nInitiator = {name}")
        print(df)

    # compute similarity between i->j and j->i
    sim = np.zeros((N,N), dtype=float)
    for i in range(N):
        for j in range(N):
            sim[i,j] = np.dot(matrix[i,j,:], matrix[j,i,:]) if i!=j else 1.0


    plt.figure(figsize=(8,6))
    im = plt.imshow(sim, interpolation='nearest')
    plt.title('Policy(Attack Actions) Similarity: Initiator vs Responder')
    plt.colorbar(im)
    plt.xticks(range(N), names, rotation=45, ha='right')
    plt.yticks(range(N), names)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    start_time = time.time()
    train_ppo(num_episodes=10000, max_steps=200)
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
