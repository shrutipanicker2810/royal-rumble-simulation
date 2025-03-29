import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython import display

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

        # Additional attributes used in attack/defend logic
        self.speed = 1.0
        self.attack_range = 30
        self.stamina = 100

        # Markov states could be something like: "idle", "attack", "defend"
        # For simplicity, let's keep them minimal:
        self.current_state = "idle"

        # Transition probabilities for MDP (example)
        # Next state probabilities given current state
        self.state_transition = {
            "idle":    {"attack": 0.4, "defend": 0.3, "idle": 0.3},
            "attack":  {"attack": 0.2, "defend": 0.4, "idle": 0.4},
            "defend":  {"attack": 0.3, "defend": 0.2, "idle": 0.5},
        }

    # -----------------------------
    # Movement
    # -----------------------------
    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def move_toward(self, target_x, target_y):
        # Calculate direction
        dx = target_x - self.x
        dy = target_y - self.y
        distance = np.sqrt(dx ** 2 + dy ** 2)
        if distance > 0:
            dx /= distance
            dy /= distance
        self.move(dx, dy)

    # -----------------------------
    # Simple MDP-based move decision
    # -----------------------------
    def decide_next_action_MDP(self):
        """Decide the wrestler's next action using a basic Markov chain."""
        transitions = self.state_transition[self.current_state]

        # Get possible actions and their probabilities
        actions, probs = zip(*transitions.items())

        # Choose the next state
        next_state = random.choices(actions, probs)[0]
        self.current_state = next_state
        return next_state

    # -----------------------------
    # Attack action
    # -----------------------------
    def attack(self, opponent):
        # Decide if we even want to "attack" or "defend" using MDP
        action = self.decide_next_action_MDP()

        # If the chosen action is "attack", proceed with actual attack
        if action == "attack":
            distance = np.sqrt((self.x - opponent.x) ** 2 + (self.y - opponent.y) ** 2)
            if distance <= self.attack_range:
                damage = random.randint(5, 10)
                opponent.health -= damage
                self.stamina -= 10
                print(f"{self.name} attacked {opponent.name}! {opponent.name}'s health is now {opponent.health}.")
            else:
                print(f"{self.name} is too far to attack {opponent.name}.")
        elif action == "defend":
            self.defend()
        else:
            # idle or do nothing
            print(f"{self.name} chooses to idle for this turn.")

    # -----------------------------
    # Defend action
    # -----------------------------
    def defend(self):
        """
        Simple defend example:
        - Gain a bit of stamina or health
        - Possibly reduce incoming damage
        """
        regain = random.randint(5, 10)
        self.stamina = min(self.stamina + regain, 100)
        print(f"{self.name} is defending, regaining {regain} stamina. Current stamina: {self.stamina}")

    def is_eliminated(self):
        return self.health <= 0

    # def attack(self, opponent):
    #     distance = np.sqrt((self.x - opponent.x) ** 2 + (self.y - opponent.y) ** 2)
    #     if distance <= self.attack_range:
    #         damage = random.randint(5, 10)  # Reduced damage
    #         opponent.health -= damage
    #         self.stamina -= 10
    #         print(f"{self.name} attacked {opponent.name}! {opponent.name}'s health is now {opponent.health}.")
    #     else:
    #         print(f"{self.name} is too far to attack {opponent.name}.")

    # def is_eliminated(self):
    #     return self.health <= 0

class WrestlingEnv(gym.Env):
    def __init__(self):
        super(WrestlingEnv, self).__init__()
        self.arena_size = 500
        self.action_space = spaces.Discrete(6)  # 0: Up, 1: Down, 2: Left, 3: Right, 4: Attack, 5: Defense
        self.observation_space = spaces.Box(
            low=0, high=self.arena_size, shape=(4,), dtype=np.float32
        )  # [x1, y1, x2, y2]

        # Initialize wrestlers
        wrestlers_data = [
            # (Name, Popularity, Height(cm), Weight(kg), Experience)
            ("John Cena", 10, 185, 114, 85),
            ("The Rock", 10, 196, 118, 90),
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

        # Choose initial players
        init_data_1 = random.choice(wrestlers_data)
        init_data_2 = random.choice([w for w in wrestlers_data if w != init_data_1])

        self.initiator = Wrestler(*init_data_1)
        self.responder = Wrestler(*init_data_2)

        # Add both wrestlers to the list
        self.wrestlers = [self.initiator, self.responder]

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.arena_size)
        self.ax.set_ylim(0, self.arena_size)
        self.ax.set_aspect('equal')

        # Simulated Annealing parameters (example usage for deciding next initiator)
        self.temperature = 1.0
        self.cooling_rate = 0.95

    def step(self, action):
        """
        Typical gym Env step function (not fully implemented):
        - action: int representing Up/Down/Left/Right/Attack/Defense
        - returns (obs, reward, done, info)
        """
        # For demonstration, we won't fully implement the step logic:
        done = False
        reward = 0

        # Perform a basic action
        if action == 0:  # Move Up
            self.initiator.move(0, 1)
        elif action == 1:  # Move Down
            self.initiator.move(0, -1)
        elif action == 2:  # Move Left
            self.initiator.move(-1, 0)
        elif action == 3:  # Move Right
            self.initiator.move(1, 0)
        elif action == 4:  # Attack
            self.initiator.attack(self.responder)
            # Check elimination
            if self.responder.is_eliminated():
                print(f"{self.responder.name} is eliminated!")
                done = True
                reward = 1  # example reward
        elif action == 5:  # Defense
            self.initiator.defend()

        # After the initiator's move, we might want to do something for the responder
        # For simplicity, let's just have the responder also attempt an action:
        # (In a real scenario, you'd have an AI or MDP for the responder's turn as well.)
        self.responder.attack(self.initiator)
        if self.initiator.is_eliminated():
            print(f"{self.initiator.name} is eliminated!")
            done = True
            reward = -1

        # Example observation: positions of initiator & responder
        obs = np.array([self.initiator.x, self.initiator.y,
                        self.responder.x, self.responder.y], dtype=np.float32)

        # Possibly apply Simulated Annealing to choose next initiator
        self.simulated_annealing_initiator_choice()

        return obs, reward, done, {}
    
    def render(self):
        self.ax.clear()  # Clear the previous plot
        self.ax.set_xlim(0, self.arena_size)
        self.ax.set_ylim(0, self.arena_size)
        self.ax.set_aspect('equal')
        # Plot wrestlers (blue circle = initiator, red circle = responder)
        for wrestler in self.wrestlers:
            if wrestler == self.initiator:
                self.ax.plot(wrestler.x, wrestler.y, 'bo', markersize=10, label=f"Initiator: {wrestler.name}")
            else:
                self.ax.plot(wrestler.x, wrestler.y, 'ro', markersize=10, label=f"Responder: {wrestler.name}")

        self.ax.legend()
        display.clear_output(wait=True)  # Clear the output for the next plot
        display.display(self.fig)  # Display the updated plot

    def reset(self):
        """
        Typically resets the environment to a start state.
        """
        # For demonstration, we won't fully implement.
        return np.zeros(4, dtype=np.float32)
    
    # --------------------------------------------
    # Simple Simulated Annealing for Next Initiator
    # --------------------------------------------
    def simulated_annealing_initiator_choice(self):
        """
        Example approach:
        - Evaluate some "score" for each wrestler
        - With certain probability (guided by temperature), pick a new initiator
        - Decrease temperature each iteration to reduce randomness over time
        """
        # Let's define a simplistic "fitness" = current health + stamina + popularity
        candidate_scores = {}
        for w in self.wrestlers:
            score = w.health + w.stamina + (w.popularity * 10)
            candidate_scores[w] = score

        # Sort wrestlers by their score
        ranked = sorted(candidate_scores, key=candidate_scores.get, reverse=True)

        best_candidate = ranked[0]
        # With some probability (based on temperature), we might pick a random candidate instead
        if random.random() < self.temperature:
            chosen_initiator = random.choice(self.wrestlers)
        else:
            chosen_initiator = best_candidate

        # Update the environment's initiator
        self.initiator = chosen_initiator
        # Responder is the other wrestler
        self.responder = [w for w in self.wrestlers if w != self.initiator][0]

        # Cool down the temperature
        self.temperature *= self.cooling_rate

def main():
    env = WrestlingEnv()
    obs = env.reset()
    done = False

    # Run a small loop of random actions for demonstration
    for _ in range(10):
        if done:
            break
        env.render()
        action = random.randint(0, 5)  # 0-5 (Up, Down, Left, Right, Attack, Defense)
        obs, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")

    plt.close()  # Close the plot when done

if __name__ == "__main__":
    main()