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

    def move(self, dx, dy):
        self.x += dx * self.speed
        self.y += dy * self.speed

    def move_toward(self, target_x, target_y):
        # Calculate the direction to move toward the target
        dx = target_x - self.x
        dy = target_y - self.y
        distance = np.sqrt(dx ** 2 + dy ** 2)
        if distance > 0:
            dx /= distance
            dy /= distance
        self.move(dx, dy)
    

    def attack(self, opponent):
        distance = np.sqrt((self.x - opponent.x) ** 2 + (self.y - opponent.y) ** 2)
        if distance <= self.attack_range:
            damage = random.randint(5, 10)  # Reduced damage
            opponent.health -= damage
            self.stamina -= 10
            print(f"{self.name} attacked {opponent.name}! {opponent.name}'s health is now {opponent.health}.")
        else:
            print(f"{self.name} is too far to attack {opponent.name}.")

    def is_eliminated(self):
        return self.health <= 0

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

        # Select any two players
        self.initiator = random.choice(wrestlers_data)
        self.responder = random.choice([wrestler for wrestler in wrestlers_data if wrestler != self.initiator])

        # Add both wrestlers to the list
        self.wrestlers = [self.initiator, self.responder]

        # self.wrestlers = [
        #     Wrestler("John Cena", 100, 100),
        #     Wrestler("The Rock", 400, 400)
        # ]
        # self.next_wrestlers = [
        #     Wrestler("Undertaker", 200, 200),
        #     Wrestler("Triple H", 300, 300),
        #     Wrestler("Randy Orton", 250, 250)
        # ]  # List of wrestlers waiting to enter

        # Matplotlib setup
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.arena_size)
        self.ax.set_ylim(0, self.arena_size)
        self.ax.set_aspect('equal')

    # def reset(self):
    #     # self.wrestlers = [
    #     #     Wrestler("John Cena", 100, 100),
    #     #     Wrestler("The Rock", 400, 400)
    #     # ]
    #     # self.next_wrestlers = [
    #     #     Wrestler("Undertaker", 200, 200),
    #     #     Wrestler("Triple H", 300, 300),
    #     #     Wrestler("Randy Orton", 250, 250)
    #     # ]
    #     return self._get_observation()

    # def step(self, action):
    #     # Wrestler 1 takes the action (if available)
    #     if len(self.wrestlers) >= 1:
    #         if action == 0:  # Up
    #             self.wrestlers[0].move(0, -1)
    #         elif action == 1:  # Down
    #             self.wrestlers[0].move(0, 1)
    #         elif action == 2:  # Left
    #             self.wrestlers[0].move(-1, 0)
    #         elif action == 3:  # Right
    #             self.wrestlers[0].move(1, 0)
    #         elif action == 4:  # Attack
    #             if len(self.wrestlers) >= 2:  # Ensure there's an opponent to attack
    #                 self.wrestlers[0].attack(self.wrestlers[1])

    #     # Wrestler 2 moves toward wrestler 1 if they are far apart (if available)
    #     if len(self.wrestlers) >= 2:
    #         distance = np.sqrt((self.wrestlers[0].x - self.wrestlers[1].x) ** 2 + (self.wrestlers[0].y - self.wrestlers[1].y) ** 2)
    #         if distance > self.wrestlers[1].attack_range:
    #             self.wrestlers[1].move_toward(self.wrestlers[0].x, self.wrestlers[0].y)
    #         else:
    #             # If close, wrestler 2 can attack or move randomly
    #             wrestler2_action = random.randint(0, 4)
    #             if wrestler2_action == 0:  # Up
    #                 self.wrestlers[1].move(0, -1)
    #             elif wrestler2_action == 1:  # Down
    #                 self.wrestlers[1].move(0, 1)
    #             elif wrestler2_action == 2:  # Left
    #                 self.wrestlers[1].move(-1, 0)
    #             elif wrestler2_action == 3:  # Right
    #                 self.wrestlers[1].move(1, 0)
    #             elif wrestler2_action == 4:  # Attack
    #                 self.wrestlers[1].attack(self.wrestlers[0])

    #     # Check if any wrestler is eliminated
    #     for i, wrestler in enumerate(self.wrestlers):
    #         if wrestler.is_eliminated():
    #             print(f"{wrestler.name} has been eliminated!")
    #             self.wrestlers.pop(i)  # Remove the eliminated wrestler
    #             if self.next_wrestlers:  # Add a new wrestler if available
    #                 new_wrestler = self.next_wrestlers.pop(0)
    #                 self.wrestlers.append(new_wrestler)
    #                 print(f"{new_wrestler.name} enters the arena!")
    #             break  # Exit the loop after removing one wrestler

    #     # Check if the match is over
    #     done = len(self.wrestlers) <= 1  # Match ends if only one wrestler remains
    #     reward = 1 if len(self.wrestlers) == 1 else 0

    #     return self._get_observation(), reward, done, {}

    # def _get_observation(self):
    #     # Return the positions of the first two wrestlers
    #     if len(self.wrestlers) >= 2:
    #         return np.array([self.wrestlers[0].x, self.wrestlers[0].y, self.wrestlers[1].x, self.wrestlers[1].y])
    #     elif len(self.wrestlers) == 1:
    #         # If only one wrestler is left, return their position twice
    #         return np.array([self.wrestlers[0].x, self.wrestlers[0].y, self.wrestlers[0].x, self.wrestlers[0].y])
    #     else:
    #         # If no wrestlers are left, return zeros (shouldn't happen in normal gameplay)
    #         return np.array([0, 0, 0, 0])

    def render(self):
        self.ax.clear()  # Clear the previous plot
        self.ax.set_xlim(0, self.arena_size)
        self.ax.set_ylim(0, self.arena_size)
        self.ax.set_aspect('equal')
        for wrestler in self.wrestlers:
            print("**************", wrestler[1])
            self.ax.plot(wrestler[1], wrestler[2], 'bo' if wrestler[0] == "John Cena" else 'ro', markersize=10, label=wrestler[0])
        self.ax.legend()
        display.clear_output(wait=True)  # Clear the output for the next plot
        display.display(self.fig)  # Display the updated plot

def main():
    env = WrestlingEnv()
    # obs = env.reset()
    done = False

    while not done:
        env.render()
        action = random.randint(0, 4)  # Random action for testing
        # obs, reward, done, info = env.step(action)
        # print(f"Reward: {reward}, Done: {done}")

    plt.close()  # Close the plot when done

if __name__ == "__main__":
    main()