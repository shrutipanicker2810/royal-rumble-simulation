import numpy as np
import pygame

class Wrestler:
    """Represents a wrestler with attributes and combat abilities."""

    height_min = 160  # will be updated dynamically
    height_max = 200
    weight_min = 50
    weight_max = 150

    def __init__(self, env, name, id, popularity, height, weight, experience):
        """Initialize a wrestler with physical attributes and stats.
        
        Args:
            env: Reference to environment
            name: Wrestler name
            id: Unique identifier
            popularity: Popularity score (0-100)
            height: Height in cm
            weight: Weight in kg
            experience: Years of experience
        """
        self._env = env
        self.name = name
        self.id = id
        self.popularity = popularity
        self.height = height
        self.weight = weight
        self.experience = experience
        self.health = 100  # Starting health
        self.max_health = 100
        self.stamina = 100  # Starting stamina
        self.max_stamina = 100
        self.wins = 0
        self.genes = np.random.uniform(0, 1, 3)  # [Strength, Agility, Defensiveness]
        self.last_action = None
        self.last_action_time = 0
        self.last_hit_time = 0
        self.match_pos = None  # Position in match array
        self._opponents = []  # List of current opponents

    def normalized_height(self):
        return 100 * (self.height - Wrestler.height_min) / max((Wrestler.height_max - Wrestler.height_min), 1)

    def normalized_weight(self):
        return 100 * (self.weight - Wrestler.weight_min) / max((Wrestler.weight_max - Wrestler.weight_min), 1)


    def compute_strength(self):
        alpha1, alpha2, alpha3, alpha4, alpha5 = 0.3, 0.1, 0.2, 0.2, 0.2
        return (
            alpha1 * self.normalized_weight()+
            alpha2 * self.normalized_height() +
            alpha3 * ((self.experience/100.0) + self.wins) +
            alpha4 * (self.popularity/10.0) +
            alpha5 * (self.health / 100.0)
        )

    def compute_stamina(self):
        beta1, beta2, beta3, beta4, beta5 = 0.3, 0.2, 0.2, 0.2, 0.1
        return (
            beta1 * self.normalized_weight() +
            beta2 * (self.experience/100.0) +
            beta3 * (self.popularity/10.0) +
            beta4 * (self.health / 100.0) +
            beta5 * self.wins
        )

    def compute_defense_rating(self):
        gamma1, gamma2 = 0.3, 0.1
        return (
            gamma1 * (self.experience/100.0) +
            gamma2 * (self.health / 100.0)
        )

    def apply_action(self, action):
        """Apply the chosen action and update wrestler state.
        
        Args:
            action: The action index to perform
        """
        self.last_action = action
        self.last_action_time = pygame.time.get_ticks()
        move_step = 0.1
        
        if not self._opponents:
            return

        # Calculate movement toward nearest opponent
        opp_pos = self._opponents[0].get_qpos()
        self_pos = self.get_qpos()
        direction = (opp_pos - self_pos) / max(np.linalg.norm(opp_pos - self_pos), 0.01)

        if action in [0, 1, 3]:  # Offensive actions
            # Move toward opponent
            new_pos = self_pos + direction * move_step
            self.set_xyz(np.array([new_pos[0], new_pos[1], 1.0]))

    def get_qpos(self):
        """Get current position in ring.
        
        Returns:
            np.array: (x,y) position
        """
        if self.match_pos is None:
            raise ValueError("Match position not set")
        pos = self._env.positions[self.match_pos]
        return np.array([pos[0], pos[1]])

    def set_xyz(self, xyz):
        """Set position in ring.
        
        Args:
            xyz: New position (x,y,z)
        """
        if self.match_pos is None:
            raise ValueError("Match position not set")
        x, y = xyz[0], xyz[1]
        self._env.positions[self.match_pos] = np.array([x, y])

    def set_match_position(self, match_pos):
        """Set position index in match array.
        
        Args:
            match_pos: Index in positions array
        """
        self.match_pos = match_pos

    def set_opponents(self, opponents):
        """Set current opponents.
        
        Args:
            opponents: List of opponent wrestlers
        """
        self._opponents = opponents