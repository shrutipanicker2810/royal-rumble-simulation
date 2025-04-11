import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random


NUM_PLAYERS = 10
MAX_RING = 3
NUM_ACTIONS = 3


MOVE_BASE_DAMAGE = {0: 10, 1: 15, 2: 25}
MOVE_BASE_STAMINA = {0: 5, 1: 8, 2: 15}

GAMMA = 0.99


class Wrestler:
    def __init__(self, name, popularity, height, weight, experience):
        self.name = name
        self.popularity = popularity  # scale 1-10
        self.height = height  # in centimeters
        self.weight = weight  # in kg
        self.experience = experience  # e.g. up to 100
        self.health = 100.0  # dynamic, 0-100
        self.stamina = 100.0  # dynamic, 0-100
        self.wins = 0  # can be used for fitness

    def compute_strength(self):
        alpha1, alpha2, alpha3, alpha4, alpha5 = 0.3, 0.1, 0.2, 0.2, 0.2
        strength_value = (
                alpha1 * self.weight +
                alpha2 * self.height +
                alpha3 * (self.experience + self.wins) +
                alpha4 * self.popularity +
                alpha5 * (self.health / 100.0)
        )
        return strength_value

    def compute_stamina_factor(self):
        beta1, beta2, beta3, beta4, beta5 = 0.3, 0.2, 0.2, 0.2, 0.1
        stamina_value = (
                beta1 * (200 - self.weight) +
                beta2 * self.experience +
                beta3 * self.popularity +
                beta4 * (self.health / 100.0) +
                beta5 * self.wins
        )
        return stamina_value

    def compute_defense(self):
        gamma1, gamma2, gamma3 = 0.3, 0.2, 0.1
        defense = gamma1 * self.experience + gamma3 * (self.health / 100.0)
        return defense

    def attack(self, opponent, action):

        if self.stamina < 10:

            return -5.0

        random_noise = random.randint(-2, 2)
        defense_val = opponent.compute_defense()
        strength_val = self.compute_strength()


        actual_damage = MOVE_BASE_DAMAGE[action] + strength_val + random_noise
        net_damage = actual_damage - (0.2 * defense_val)
        if net_damage < 0:
            net_damage = 0.0
        opponent.health = max(0.0, opponent.health - net_damage)


        stamina_factor = self.compute_stamina_factor()
        actual_stamina = MOVE_BASE_STAMINA[action] + stamina_factor + random_noise
        net_stamina = actual_stamina - (0.2 * defense_val)
        if net_stamina < 0:
            net_stamina = 5.0
        self.stamina = max(0.0, self.stamina - net_stamina)


        reward = net_damage - 0.5 * net_stamina
        return reward



class WrestlingEnv:


    def __init__(self, players):
        self.all_players = players
        self.waiting_list = players.copy()
        self.ring = []
        self.rl_agent = self.waiting_list[0]
        self.waiting_list.remove(self.rl_agent)
        self.steps_since_last_entry = 0
        self.entry_interval = 3

    def reset(self):

        for p in self.all_players:
            p.health = 100.0
            p.stamina = 100.0

        self.ring = [self.rl_agent]
        self.waiting_list = [p for p in self.all_players if p != self.rl_agent]
        if self.waiting_list:
            opp = random.choice(self.waiting_list)
            self.ring.append(opp)
            self.waiting_list.remove(opp)
        self.steps_since_last_entry = 0
        return self._get_pair_state()

    def _get_pair_state(self):

        if self.rl_agent not in self.ring or len(self.ring) < 2:
            return None
        opponents = [p for p in self.ring if p != self.rl_agent]
        opponent = random.choice(opponents)
        index_i = self.all_players.index(self.rl_agent)
        index_r = self.all_players.index(opponent)
        state = np.array([
            index_i / (NUM_PLAYERS - 1),
            self.rl_agent.health / 100.0,
            self.rl_agent.stamina / 100.0,
            self.rl_agent.experience / 100.0,
            self.rl_agent.popularity / 10.0,
            index_r / (NUM_PLAYERS - 1),
            opponent.health / 100.0,
            opponent.stamina / 100.0,
            opponent.experience / 100.0,
            opponent.popularity / 10.0,
        ], dtype=np.float32)
        return state

    def step(self, action):

        reward_total = 0.0
        self.steps_since_last_entry += 1
        if self.steps_since_last_entry >= self.entry_interval and len(self.ring) < MAX_RING and self.waiting_list:
            new_opp = random.choice(self.waiting_list)
            self.ring.append(new_opp)
            self.waiting_list.remove(new_opp)
            self.steps_since_last_entry = 0

        if len(self.ring) == 1:
            return None, reward_total, True, {}

        pair = random.sample(self.ring, 2)
        initiator = random.choice(pair)
        responder = pair[0] if pair[1] == initiator else pair[1]

        if initiator == self.rl_agent:
            current_action = action
        else:
            current_action = random.choice([0, 1, 2])

        r = initiator.attack(responder, current_action)
        if initiator == self.rl_agent:
            reward_total += r

        removed = []
        for fighter in pair:
            if fighter.health <= 0:
                removed.append(fighter)
        for fighter in removed:
            if fighter in self.ring:
                self.ring.remove(fighter)

        done = len(self.ring) <= 1
        next_state = None
        if self.rl_agent in self.ring and len(self.ring) >= 2:
            next_state = self._get_pair_state()

        return next_state, reward_total, done, {}



def get_state(initiator, responder, index_i, index_r):

    state = np.array([
        index_i / (NUM_PLAYERS - 1),
        initiator.health / 100.0,
        initiator.stamina / 100.0,
        initiator.experience / 100.0,
        initiator.popularity / 10.0,
        index_r / (NUM_PLAYERS - 1),
        responder.health / 100.0,
        responder.stamina / 100.0,
        responder.experience / 100.0,
        responder.popularity / 10.0,
    ], dtype=np.float32)
    return state



def build_policy_network(state_dim, action_dim):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(state_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(action_dim, activation='softmax')
    ])
    return model



def compute_returns(rewards, gamma=GAMMA):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = np.array(returns, dtype=np.float32)
    if returns.std() > 0:
        returns = (returns - returns.mean()) / (returns.std())
    return returns

def create_players():
    players_data = [
        ("W1", 8, 185, 100, 80),
        ("W2", 7, 175, 90, 70),
        ("W3", 9, 190, 105, 85),
        ("W4", 6, 180, 95, 65),
        ("W5", 8, 185, 100, 75),
        ("W6", 7, 170, 85, 60),
        ("W7", 9, 195, 110, 90),
        ("W8", 6, 175, 88, 68),
        ("W9", 8, 182, 98, 72),
        ("W10", 7, 178, 92, 66)
    ]
    return [Wrestler(*data) for data in players_data]


def train_policy_gradient_full_round(num_episodes=300, learning_rate=0.001):
    players = create_players()
    env = WrestlingEnv(players)
    state_dim = 10  # Pairwise state vector dimension
    action_dim = NUM_ACTIONS
    policy_net = build_policy_network(state_dim, action_dim)
    optimizer = keras.optimizers.Adam(learning_rate)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        transitions = []
        total_reward_episode = 0.0

        while not done:
            if state is None:
                break
            state_tensor = tf.expand_dims(state, axis=0)
            action_probs = policy_net(state_tensor, training=False)
            action_probs_np = action_probs.numpy().squeeze()
            action = np.random.choice(action_dim, p=action_probs_np)
            log_prob = tf.math.log(action_probs[0, action] + 1e-10)
            next_state, reward, done, info = env.step(action)
            transitions.append((state, action, reward))
            total_reward_episode += reward
            state = next_state

        rewards = [t[2] for t in transitions]
        returns = compute_returns(rewards, gamma=GAMMA)

        with tf.GradientTape() as tape:
            total_loss = 0.0
            for (s, a, _), G_val in zip(transitions, returns):
                s_tensor = tf.expand_dims(s, axis=0)
                probs = policy_net(s_tensor, training=True)
                log_prob = tf.math.log(probs[0, a] + 1e-10)
                total_loss += -log_prob * G_val
            total_loss = total_loss / len(transitions)
        grads = tape.gradient(total_loss, policy_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))

        if episode % 10 == 0:
            print(
                f"Episode {episode:4d}: Total Reward = {total_reward_episode:.3f}, Avg Loss = {total_loss.numpy():.3f}")

    return policy_net


def test_policy_full_round(policy_net, use_three=False):
    players = create_players()
    env = WrestlingEnv(players)
    for w in players:
        w.health = 100.0
        w.stamina = 100.0
    if use_three:
        test_players = random.sample(players, 3)
        print("\n--- Testing with 3 wrestlers in the ring ---")
    else:
        test_players = players
        print("\n--- Testing with pairwise interactions ---")
    state = env.reset()
    done = False
    step_counter = 0
    while not done:
        if state is None:
            break
        state_tensor = tf.expand_dims(state, axis=0)
        action_probs = policy_net(state_tensor, training=False).numpy().squeeze()
        action = np.argmax(action_probs)
        action_str = ["Punch", "Kick", "Signature"][action]
        print(f"Step {step_counter}: State: {state}, Action: {action_str}, Probabilities: {action_probs}")
        next_state, reward, done, info = env.step(action)
        print(f"    Reward: {reward:.3f}")
        state = next_state
        step_counter += 1
    if env.rl_agent in env.ring:
        print(f"\nRound ended: RL agent {env.rl_agent.name} is the last standing!")
    else:
        print(f"\nRound ended: RL agent {env.rl_agent.name} was eliminated!")



def generate_pairwise_matrix(policy_net):

    players = create_players()
    matrix = [[None for _ in range(NUM_PLAYERS)] for _ in range(NUM_PLAYERS)]
    for i in range(NUM_PLAYERS):
        for j in range(NUM_PLAYERS):
            if i == j:
                matrix[i][j] = "N/A"
            else:
                for p in players:
                    p.health = 100.0
                    p.stamina = 100.0
                state = get_state(players[i], players[j], i, j)
                state_tensor = tf.expand_dims(state, axis=0)
                action_probs = policy_net(state_tensor, training=False).numpy().squeeze()
                matrix[i][j] = action_probs.tolist()
    return matrix



if __name__ == "__main__":
    print("Training Policy Gradient Agent for Full Round Wrestling...")
    trained_policy = train_policy_gradient_full_round(num_episodes=300, learning_rate=0.001)
    print("\nTraining complete.\n")

    test_policy_full_round(trained_policy, use_three=True)

    print("\nGenerating 10x10 Pairwise Interaction Probability Matrix:")
    pairwise_matrix = generate_pairwise_matrix(trained_policy)
    for i in range(NUM_PLAYERS):
        for j in range(NUM_PLAYERS):
            print(f"W{i + 1} vs W{j + 1}: {pairwise_matrix[i][j]}")
        print()
