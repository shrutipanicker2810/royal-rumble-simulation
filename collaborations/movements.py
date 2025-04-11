"""
This is the backend code that sets up the wrestling environment using gym.
1. It selects the first initiator through Simulated Annealing
2. Picks an action from punch, kick, SM, stay idle
3. If the environment's responder is eliminated after an action (reward[initiator] += 100), it adds a new wrestler to the arena.
4. Builds the logic of attack and defense functions
4.1 If wrestler chooses stay idle, regain stamina
4.2 If wrestler chooses attack and has low stamina, stay idle and regain stamina
4.2 If wrestler chooses attack and has enough stamina, attack the opponent and calculate opponent's defense, attacker's stamina and attacker's strength
5 Calculate the reward for the attacker and responder at each step (end of the action)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython import display
import pygame
from variables import *

class Wrestler:
    def __init__(self, name, popularity, height, weight, experience):
        self.name = name
        self.x = 250
        self.y = 0
        self.popularity = popularity
        self.height = height
        self.weight = weight
        self.experience = experience
        self.health = 100
        self.wins = 0
        self.stamina = 100
        self.reward = 0
        self.elimination_count = 0

        self.attack_range = 500 # setting to full arena for now


    # -----------------------------
    # Movement
    # -----------------------------
    def move(self, dx, dy):
        self.x += dx
        self.y += dy


    def move_toward(self, target_x, target_y):
        dx = target_x - self.x
        dy = target_y - self.y
        distance = np.sqrt(dx ** 2 + dy ** 2)
        if distance > 0:
            dx /= distance
            dy /= distance
        self.move(dx, dy)


    # -----------------------------
    # Compute strength, stamina and defense
    # -----------------------------
    def compute_strength(self, wrestler):
        alpha1, alpha2, alpha3, alpha4, alpha5 = 0.3, 0.1, 0.2, 0.2, 0.2
        strength_value = (
            alpha1 * wrestler.weight +
            alpha2 * wrestler.height +
            alpha3 * (wrestler.experience + wrestler.wins) +
            alpha4 * wrestler.popularity +
            alpha5 * (wrestler.health / 100.0)
        )
        return strength_value
    

    def compute_stamina(self, wrestler):
        beta1, beta2, beta3, beta4, beta5 = 0.3, 0.2, 0.2, 0.2, 0.1
        stamina_value = (
            beta1 * (200 - wrestler.weight) +
            beta2 * wrestler.experience +
            beta3 * wrestler.popularity +
            beta4 * (wrestler.health / 100.0) +
            beta5 * wrestler.wins
        )
        return stamina_value


    def compute_defense_rating(self, wrestler):
        gamma1, gamma2, gamma3 = 0.3, 0.2, 0.1
        return (gamma1 * wrestler.experience +
            gamma3 * (wrestler.health / 100.0))


    # -----------------------------
    # Idle action
    # -----------------------------
    def stay_idle(self):
        """
        Simple idle example:
        - Gain a bit of stamina or health
        - Possibly reduce incoming damage
        """
        regain = random.randint(5, 10)
        self.stamina = min(self.stamina + regain, 100)
        print(f"{self.name} is not attacking, regaining {regain} stamina. Current stamina: {self.stamina}")


    # -----------------------------
    # Attack action
    # -----------------------------
    def attack(self, opponent, action):
        '''Turn-Based
        You designate one wrestler as the active initiator each turn.
        That initiator chooses an action (attack a specific opponent, defend, etc.).
        Only the initiator and chosen responder get immediate non-zero rewards.
        Then you move on to the next wrestler's turn.'''
        if self.stamina < 10:  # not enough stamina to attack
            print(f"{self.name} is too tired to attack!")
            self.stamina += 5 
            self.stay_idle()
            return

        # distance = np.sqrt((self.x - opponent.x) ** 2 + (self.y - opponent.y) ** 2)
        # if distance > self.attack_range:
        #     print(f"{self.name} is too far to attack {opponent.name}.")
        #     return
        
        random_noise = random.randint(-2, 2)
        # 0) Calculate opponent's defense
        defense_val = self.compute_defense_rating(opponent)

        # 1) Calculate opponent's damage
        strength_factor = self.compute_strength(self)
        actual_damage = MOVE_BASE_DAMAGE[action] + strength_factor + random_noise

        net_damage = actual_damage - (0.2 * defense_val)
        if net_damage < 0:
            net_damage = 0

        opponent.health -= net_damage

        # 2) Calculate attacker's stamina
        stamina_factor = self.compute_stamina(self)
        actual_stamina = MOVE_BASE_STAMINA[action] + stamina_factor + random_noise

        net_stamina = actual_stamina - (0.2 * defense_val)
        if net_stamina <0:
            net_stamina = 5

        self.stamina -= net_stamina

        # (a) Reward for attacker's Damage dealing
        self.reward += net_damage 
        # (b) Penalty for attacker's Stamina Usage
        self.reward -= 0.5 * net_stamina
        # (c) Penalty for opponent's Damage dealing
        opponent.reward -= net_damage

        print(f"{self.name} attacked {opponent.name}! {opponent.name}'s health is now {opponent.health} and {self.name}'s stamina is now {self.stamina}")
        print(f" Reward for {self.name} is {self.reward} and reward for {opponent.name} is {opponent.reward}")
       

    def is_eliminated(self):
        return self.health <= 0

class WrestlingEnv(gym.Env):
    def __init__(self):
        super(WrestlingEnv, self).__init__()
        self.arena_size = 500
        self.elimination_count = 0
        self.eliminated_names = set()
        self.action_space = spaces.Discrete(8)  # 0: Up, 1: Down, 2: Left, 3: Right, 4: Kicks, 5: Punches, 6: Signature moves, 7: stay_idle 
        # remove 0-3 from action space in case we don't need spatial awareness

        self.observation_space = spaces.Box(
            low=np.array([0,    0,  0,  0,   0,    0,  0,  0,    0,     0,   0,  0]),
            high=np.array([100, 100, 3,  1,  100,  100, 3,  1, self.arena_size, self.arena_size, self.arena_size, self.arena_size]),
            dtype=np.float32
        ) # [health_initiator, stamina_initiatior, last_move_initiator, ad_initiator,
        # health_respnder, stamina_responder, last_move_responder, ad_responder,
        # x_initiator, y_initiator, x_responder, y_responder] # Remove x and y if the decisions does not depend on the spatial awareness

        # Initialize wrestlers
        self.wrestlers_data = [
            # (Name, Popularity, Height(cm), Weight(kg), Experience)
            ("John Cena", 10, 185, 114, 85),
            ("The Great Khali", 10, 196, 118, 90),
            ("The Undertaker", 9, 208, 136, 95),
            ("Triple H", 9, 193, 116, 88),
            ("Randy Orton", 8, 196, 113, 82),
            ("Brock Lesnar", 9, 191, 130, 78),
            ("Roman Reigns", 9, 191, 120, 75),
            ("Seth Rollins", 8, 185, 98, 72),
            ("Stone Cold Steve Austin", 10, 188, 114, 92),
            ("Hulk Hogan", 9, 201, 137, 95),
            ("Ric Flair", 8, 185, 110, 98),
            ("Shawn Michaels", 8, 185, 102, 90),
            ("Bret Hart", 7, 183, 106, 85),
            ("AJ Styles", 8, 180, 99, 80),
            ("Edge", 8, 193, 109, 82),
            ("Chris Jericho", 8, 183, 103, 88),
            ("Big Show", 7, 213, 200, 85),
            ("Kane", 8, 208, 147, 82),
            ("Rey Mysterio", 9, 168, 79, 90),
            ("Batista", 8, 198, 130, 75),
            ("CM Punk", 7, 185, 98, 72),
            ("Kevin Owens", 7, 183, 122, 68),
            ("Finn Bálor", 7, 180, 86, 70),
            ("Becky Lynch", 8, 168, 61, 65),
            ("Charlotte Flair", 8, 180, 72, 65),
            ("Ronda Rousey", 8, 168, 61, 60),
            ("Macho Man Randy Savage", 7, 188, 112, 85),
            ("André the Giant", 9, 224, 240, 90),
            ("Dwayne 'The Rock' Johnson", 10, 196, 118, 90),
            ("Chyna", 7, 178, 82, 70),
            ("Mick Foley", 7, 188, 130, 85),
            ("Bray Wyatt", 7, 191, 129, 65)
        ]

        init_data_1 = random.choice(self.wrestlers_data)
        init_data_2 = random.choice([w for w in self.wrestlers_data if w != init_data_1])

        self.initiator = Wrestler(*init_data_1)
        self.responder = Wrestler(*init_data_2)

        # Add both wrestlers to the list
        self.wrestlers = [self.initiator, self.responder]

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.arena_size)
        self.ax.set_ylim(0, self.arena_size)
        self.ax.set_aspect('equal')

        self.temperature = 1.0
        self.cooling_rate = 0.95


    def calculate_fitness(self, wrestler):
            return (
                0.4 * (wrestler.health / 100) +
                0.3 * (wrestler.stamina / 100) +
                0.2 * wrestler.experience +
                0.1 * wrestler.popularity +
                0.05 * wrestler.wins
        )
    

    def add_wrestlers(self):
        # Add a new wrestler into the arena based on fitness score,
        # prioritizing the best-fit unused wrestler.
        current_names = [w.name for w in self.wrestlers]
        available = [w for w in self.wrestlers_data if w[0] not in current_names and w[0] not in self.eliminated_names]

        if not available:
            print("No more wrestlers to add.")
            return

        # Convert to Wrestler instances temporarily to calculate fitness
        candidates = [Wrestler(*data) for data in available]
        candidates_fitness = {w: self.calculate_fitness(w) for w in candidates}

        # Sort by fitness
        sorted_candidates = sorted(candidates_fitness.items(), key=lambda x: x[1], reverse=True)

        # With SA randomness
        if random.random() < self.temperature:
            chosen = random.choice(candidates)
        else:
            chosen = sorted_candidates[0][0]

        self.wrestlers.append(chosen)
        print(f"[SA] New wrestler joined: {chosen.name}")


    def simulated_annealing_initiator_choice(self):
    # Use Simulated Annealing to select the initiator based on fitness score
    # with some randomness that decreases over time.
        if len(self.wrestlers) < 2:
            print("Not enough wrestlers for selection.")
            return

        # Score all wrestlers
        candidate_scores = {w: self.calculate_fitness(w) for w in self.wrestlers}
        best_candidate = max(candidate_scores, key=candidate_scores.get)

        # With probability based on temperature, pick random instead of best
        if random.random() < self.temperature:
            self.initiator = random.choice(self.wrestlers)
        else:
            self.initiator = best_candidate

        # Pick responder randomly from the rest
        others = [w for w in self.wrestlers if w != self.initiator]
        self.responder = random.choice(others)

        print(f"[SA] Initiator: {self.initiator.name}, Responder: {self.responder.name}, Temp: {self.temperature:.3f}")

        # Cool down the temperature after each selection
        self.temperature *= self.cooling_rate

        # 🔁 Reset temperature when too low
        if self.temperature < 0.01:
            self.temperature = 1.0


    def handle_elimination(self, eliminated_wrestler):  
        print(f"{eliminated_wrestler.name} is eliminated and removed from the arena.")
        self.eliminated_names.add(eliminated_wrestler.name)
        self.wrestlers.remove(eliminated_wrestler)

        available_wrestlers = [
            w for w in self.wrestlers_data
            if w[0] not in [wr.name for wr in self.wrestlers] and w[0] not in self.eliminated_names
        ]

        if available_wrestlers:
            new_wrestler_data = random.choice(available_wrestlers)
            new_wrestler = Wrestler(*new_wrestler_data)
            self.wrestlers.append(new_wrestler)
            print(f"New wrestler entered: {new_wrestler.name}")
            
            self.simulated_annealing_initiator_choice()
        else:
            print("No wrestlers left to enter the arena.")


    def step(self, action):
        """
        Gym Env step function
        - action: int representing Up/Down/Left/Right/3 Attacks/stay_idle
        - returns (obs, reward, done, info)
        """
        done = False # base condition when 29 players are eliminated and only one remains
        elimination_count = 0

        if elimination_count >= 29:
            done = True

        # if action == 0:  # Move Up
        #     self.initiator.move(0, 1)
        # elif action == 1:  # Move Down
        #     self.initiator.move(0, -1)
        # elif action == 2:  # Move Left
        #     self.initiator.move(-1, 0)
        # elif action == 3:  # Move Right
        #     self.initiator.move(1, 0)

        self.initiator.attack(self.responder, action)
        if self.responder.is_eliminated():
                self.initiator.reward += 10
                self.elimination_count += 1
                self.handle_elimination(self.responder)
                # done = True
                # Bring more wrestlers into the game irrespective of time TODO: Rajkesh
                # function to add wrestlers to the list
        elif action == 7:  # idle
            self.initiator.stay_idle()

        # TODO: VS increase stamina for everyone in the arena

        obs = np.array([self.initiator.health, self.initiator.stamina, action, 1 , self.responder.health, self.responder.stamina, 7, 1 , 
                        self.initiator.x, self.initiator.y, self.responder.x, self.responder.y], dtype=np.float32)
        return obs, self.initiator.reward, done, {}

    def reset(self):
        """
        Typically resets the environment to a start state.
        """
        return np.zeros(12, dtype=np.float32)


def main():
    pygame.init()
    env = WrestlingEnv(ring_size=3.0, entry_interval=5)
    obs = env.reset()
    done = False

    env.simulated_annealing_initiator_choice()

    while not done:
        env.render()
        action = random.randint(4, 6)  # 4-6: attacks
        obs, reward, done, info = env.step(action)

        if env.responder.is_eliminated():
            env.initiator.reward += 10  # Bonus for elimination
            env.add_wrestlers()

        if done:
            env.initiator.reward += 100  # Bonus for last man standing
            break

    plt.close()

if __name__ == "__main__":
    main()