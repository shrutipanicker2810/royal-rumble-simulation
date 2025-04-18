import pygame
import numpy as np
import random
import math
import sys
from wrestler import Wrestler
from agent import WrestlingAgent
from wrestling_viz import WrestlingViz
from genetic_evolution import evolve_wrestlers

class BattleRoyaleEnv:
    """Environment for a wrestling battle royale with multiple wrestlers entering over time.
    
    Attributes:
        WIN_REWARD (float): Reward for eliminating an opponent
        MIN_SEPARATION (float): Minimum distance between wrestlers when entering
    """
    
    WIN_REWARD = 500.0
    MIN_SEPARATION = 2  # Minimum distance between wrestlers

    def __init__(self, ring_size=5.0, entry_interval=10):
        """Initialize the battle royale environment.

        Args:
            ring_size (float): Size of the wrestling ring
            entry_interval (int): Timesteps between new wrestler entries
        """
        self.ring_size = ring_size
        self.entry_interval = entry_interval
        self.entry_timer = 0
        self.wrestlers = []  # All wrestlers in the match
        self.active_wrestlers = []  # Currently active wrestlers
        self.eliminated_wrestlers = []  # Wrestlers who have been eliminated
        self.positions = []  # Positions of all wrestlers
        self.last_special_entry = -self.entry_interval - 1  # Timestep of last forced entry
        # Simulated annealing parameters for initiator selection
        self.temperature = 1.0  # Initial temperature
        self.cooling_rate = 0.95  # Cooling rate per selection
        self.current_initiator = None  # Track the current initiator for SA
        self.current_initiator_fitness = None  # Track the fitness of the current initiator

    def reset(self):
        """Reset the environment to initial state."""
        self.entry_timer = 0
        self.active_wrestlers = []
        self.eliminated_wrestlers = []
        self.positions = []
        self.last_special_entry = -self.entry_interval - 1
        # Reset SA parameters
        self.temperature = 1.0
        self.current_initiator = None
        self.current_initiator_fitness = None
        return self._get_obs()

    def step(self, actions):
        """Advance the simulation by one timestep.
        
        Args:
            actions (dict): Actions for each active wrestler
            
        Returns:
            tuple: (observations, rewards, dones, infos, initiator, responder)
        """
        self.entry_timer += 1
        print(f"\n Timestep {self.entry_timer}: Active Wrestlers = {len(self.active_wrestlers)}")

        # Determine if we should add a new wrestler
        can_add_regular = (self.entry_timer % self.entry_interval == 0 and 
                         self.entry_timer - self.last_special_entry >= self.entry_interval)
        must_add_special = (len(self.active_wrestlers) == 1)  # Force add if only 1 wrestler left
        has_available_wrestlers = len(self.wrestlers) > len(self.active_wrestlers) + len(self.eliminated_wrestlers)

        if has_available_wrestlers and (can_add_regular or must_add_special):
            self._add_new_wrestler()
            if must_add_special:
                self.last_special_entry = self.entry_timer
                print(f"Special entry triggered at timestep {self.entry_timer}")

        # Initialize rewards, done flags and info dicts
        rewards = {w.id: 0 for w in self.active_wrestlers}
        dones = {w.id: False for w in self.active_wrestlers}
        infos = {w.id: {} for w in self.active_wrestlers}

        # Select two wrestlers to interact
        initiator, responder = self._select_combatants()

        if initiator and responder:
            action = actions.get(initiator.id, 2)  # Default to no-op if no action specified
            if initiator.stamina <= 0:
                action = 2
            attack_types = {0: "Punch", 1: "Kick", 2: "Signature", 3: "No-op"}
            attack_type = attack_types.get(action, "Unknown")

            print(f"Initiator: {initiator.name} - Health: {initiator.health:.1f} - Attack Type: {attack_type}")

            # Move initiator toward responder (with increased speed)
            init_pos = self.positions[initiator.match_pos]
            resp_pos = self.positions[responder.match_pos]
            direction = (resp_pos - init_pos) / max(np.linalg.norm(resp_pos - init_pos), 0.01)
            new_pos = init_pos + direction * 0.5  # Increased movement speed
            new_pos = np.clip(new_pos, -self.ring_size, self.ring_size)
            self.positions[initiator.match_pos] = new_pos

            initiator.apply_action(action)

            if action in [0, 1, 2]:  # If attack action
                if self._check_hit(initiator, responder):
                    # Calculate damage based on attack type
                    random_noise = random.randint(-5, 5)
                    damage = {
                        0: initiator.compute_strength() * 0.3,  # Punch
                        1: initiator.compute_strength() * 0.4,  # Kick
                        2: initiator.compute_strength() * 0.6   # Signature move
                    }[action]
                    stamina = {
                        0: initiator.compute_stamina() * 0.2,  # Punch
                        1: initiator.compute_stamina() * 0.3,  # Kick
                        2: initiator.compute_stamina() * 0.4   # Signature move
                    }[action]
                    defense_val = responder.compute_defense_rating()
                    raw_damage = damage + random_noise - (0.2 * defense_val)
                    # Apply damage and clamp health so it doesn't go below 0:
                    responder.health = max(0, responder.health - raw_damage)

                    #responder.health -= (damage + random_noise - (0.2 * defense_val))
                    initiator.stamina -= (stamina + random_noise - (0.2 * defense_val))
                    rewards[initiator.id] = rewards[initiator.id] + (random_noise + damage - (0.5 * stamina ) - (0.2 * defense_val)) 
                    responder.last_hit_time = pygame.time.get_ticks()
                    print(f"Rewards Gained by {initiator.name}: {rewards[initiator.id]:.1f}")
                    print(f"Responder: {responder.name} - Health after defending: {responder.health:.1f}")

                    if responder.health <= 0:  # Elimination check
                        self._handle_elimination(responder)
                        rewards[initiator.id] += self.WIN_REWARD
                        dones[responder.id] = True
                        infos[initiator.id]["win"] = True
                        infos[responder.id]["lose"] = True
                        print(f"Updated Rewards for {initiator.name} after elimination: {rewards[initiator.id]:.1f}")
                else:
                    print(f"Attack missed! Distance too far.")
                    print(f"Responder: {responder.name} - Health after defending: {responder.health:.1f}")
            else:
                initiator.stamina = min(initiator.max_stamina, initiator.stamina + 5)
                print(f"Rewards Gained by {initiator.name}: {rewards[initiator.id]:.1f}")
                print(f"Responder: {responder.name} - Health after defending: {responder.health:.1f}")
            
        else:
            print("No combat this timestep (less than 2 wrestlers).")

        self.latest_rewards = rewards
        self._update_simulation()

        # Check if match is over (only 1 wrestler left and no more to enter)
        if len(self.active_wrestlers) <= 1 and len(self.wrestlers) == len(self.active_wrestlers) + len(self.eliminated_wrestlers):
            for w in self.active_wrestlers:
                dones[w.id] = True
                infos[w.id]["winner"] = True

        return self._get_obs(), rewards, dones, infos, initiator, responder

    def _add_new_wrestler(self):
        """Add a new wrestler to the active match with random position."""
        available = [w for w in self.wrestlers if w not in self.active_wrestlers and w not in self.eliminated_wrestlers]
        if available:
            new_wrestler = available[0]
            pos_idx = len(self.positions)
            
            # Try to find a valid position not too close to others
            max_attempts = 10
            for _ in range(max_attempts):
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(0.5, self.ring_size * 0.8)
                candidate_pos = np.array([radius * math.cos(angle), radius * math.sin(angle)])
                
                # Check distance to other wrestlers
                too_close = False
                for other_wrestler in self.active_wrestlers:
                    other_pos = self.positions[other_wrestler.match_pos]
                    if np.linalg.norm(candidate_pos - other_pos) < self.MIN_SEPARATION:
                        too_close = True
                        break
                
                if not too_close:
                    self.positions.append(candidate_pos)
                    break
            else:
                # If no valid position found after max attempts, place at center
                self.positions.append(np.array([0.0, 0.0]))
            
            # Add wrestler to active list and update opponent tracking
            self.active_wrestlers.append(new_wrestler)
            new_wrestler.set_match_position(pos_idx)
            new_wrestler.set_opponents([w for w in self.active_wrestlers if w != new_wrestler])
            for w in self.active_wrestlers:
                if new_wrestler not in w._opponents:
                    w._opponents.append(new_wrestler)
            print(f"Added {new_wrestler.name} at position {self.positions[pos_idx]}. Active Wrestlers: {len(self.active_wrestlers)}")

    def calculate_fitness(self, wrestler):
        """Calculate the fitness score of a wrestler based on health, stamina, experience, and popularity."""
        return (0.4 * (wrestler.health / wrestler.max_health) +
                0.3 * (wrestler.stamina / wrestler.max_stamina) +
                0.2 * (wrestler.experience / 10) +
                0.1 * (wrestler.popularity / 100))
    
    def _select_combatants(self):
        """Select two wrestlers to interact using simulated annealing for the initiator."""
        if len(self.active_wrestlers) < 2:
            return None, None

        # Simulated annealing to select the initiator
        max_iterations = 10  # Number of iterations for SA within this selection
        if self.current_initiator is None or self.current_initiator not in self.active_wrestlers:
            # Initialize the current initiator if not set or if it was eliminated
            self.current_initiator = random.choice(self.active_wrestlers)
            self.current_initiator_fitness = self.calculate_fitness(self.current_initiator)

        # Perform simulated annealing iterations
        for _ in range(max_iterations):
            # Select a new candidate initiator randomly
            candidate = random.choice(self.active_wrestlers)
            candidate_fitness = self.calculate_fitness(candidate)

            # Calculate the difference in fitness (delta)
            delta = candidate_fitness - self.current_initiator_fitness

            # Always accept better candidates; accept worse ones with probability based on temperature
            if delta > 0 or random.random() < math.exp(delta / max(self.temperature, 0.01)):
                self.current_initiator = candidate
                self.current_initiator_fitness = candidate_fitness

            # Cool down the temperature after each iteration
            self.temperature *= self.cooling_rate

        # Reset temperature if it becomes too low
        if self.temperature < 0.01:
            self.temperature = 1.0

        initiator = self.current_initiator
        print(f"[SA] Initiator: {initiator.name}, Fitness: {self.current_initiator_fitness:.3f}, Temp: {self.temperature:.3f}")

        # Select random responder
        possible_responders = [w for w in self.active_wrestlers if w != initiator]
        responder = random.choice(possible_responders) if possible_responders else None
        return initiator, responder

    def _check_hit(self, attacker, defender):
        """Check if an attack hits based on distance between wrestlers."""
        dist = np.linalg.norm(self.positions[attacker.match_pos] - self.positions[defender.match_pos])
        print(f"Distance between {attacker.name} and {defender.name}: {dist:.2f}")
        return dist < 1.5  # Hit if within range

    def _handle_elimination(self, wrestler):
        """Remove a wrestler from active competition."""
        self.eliminated_wrestlers.append(wrestler)
        self.active_wrestlers.remove(wrestler)
        # Remove from opponents lists
        for w in self.active_wrestlers:
            if wrestler in w._opponents:
                w._opponents.remove(wrestler)
        print(f"Eliminated {wrestler.name}. Active Wrestlers: {len(self.active_wrestlers)}")

    def _update_simulation(self):
        """Update wrestler positions and handle collisions."""
        for w in self.active_wrestlers:
            pos = self.positions[w.match_pos]
            new_pos = pos.copy()

            # Initial movement toward center for first few timesteps
            if self.entry_timer <= 2 and w not in self.eliminated_wrestlers:
                direction = -pos / max(np.linalg.norm(pos), 0.01)
                new_pos = pos + direction * 0.5
            elif w.last_action != 3:  # If not doing no-op
                new_pos = pos

            # Calculate repulsion from other wrestlers to prevent crowding
            repulsion_force = np.zeros(2)
            for other_wrestler in self.active_wrestlers:
                if other_wrestler == w:
                    continue
                other_pos = self.positions[other_wrestler.match_pos]
                dist = np.linalg.norm(new_pos - other_pos)
                if dist < self.MIN_SEPARATION and dist > 0:
                    direction = (new_pos - other_pos) / dist
                    repulsion_magnitude = (self.MIN_SEPARATION - dist) * 0.1
                    repulsion_force += direction * repulsion_magnitude

            new_pos += repulsion_force
            new_pos = np.clip(new_pos, -self.ring_size, self.ring_size)
            self.positions[w.match_pos] = new_pos

    def _get_obs(self):
        """Get observations for each active wrestler."""
        obs = {}
        for w in self.active_wrestlers:
            self_pos = self.positions[w.match_pos]
            # Get position of nearest opponent
            opp_pos = min(w._opponents, key=lambda o: np.linalg.norm(self.positions[o.match_pos] - self_pos)).get_qpos() if w._opponents else np.zeros(2)
            obs[w.id] = np.concatenate([np.array([w.health, w.stamina, w.last_action or 0]), self_pos, opp_pos])
        return obs
    
# def filter_wrestlers_by_lineage(wrestlers):
#     """Filter wrestlers to ensure only one per lineage, prioritizing initial versions."""
#     lineage_dict = {}
#     for wrestler in wrestlers:
#         # Extract base name (lineage) by removing "-Evolved-#" suffix
#         base_name = wrestler.name.split("-Evolved-")[0]
#         if base_name not in lineage_dict:
#             lineage_dict[base_name] = wrestler
#         else:
#             # Prioritize initial version (no "-Evolved-" in name) or lowest ID
#             current = lineage_dict[base_name]
#             if "-Evolved-" not in wrestler.name and "-Evolved-" in current.name:
#                 lineage_dict[base_name] = wrestler
#             elif "-Evolved-" in wrestler.name and "-Evolved-" not in current.name:
#                 continue
#             elif wrestler.id < current.id:
#                 lineage_dict[base_name] = wrestler
#     return list(lineage_dict.values())
    
def run_battle_royale(num_generations=1):
    """Run the battle royale simulation with visualization."""
    pygame.init()
    env = BattleRoyaleEnv(ring_size=3.0, entry_interval=5)  # Faster entry than original (20->5)
    viz = WrestlingViz()
    
    # Create wrestlers with stats (name, popularity, height, weight, experience)
    wrestlers_data = [
    # (Name, Popularity, Height(cm), Weight(kg), Experience (out of 100))
    # Top-tier wrestlers
    ("Roman R", 10, 191, 120, 88),
    ("Brock L", 9, 191, 130, 95),
    ("Seth R", 9, 185, 98, 80),
    ("Becky L", 9, 168, 61, 77),
    # Moderate-tier wrestlers
    ("Finn B", 7, 180, 86, 65),
    ("Kevin O", 7, 183, 122, 70),
    ("AJ Styles", 7, 180, 99, 73),
    # Low-tier wrestlers
    ("Dominik M", 5, 185, 91, 49),
    ("Liv M", 5, 160, 50, 37),
    ("Otis", 4, 178, 150, 18)
    ]
    
    env.wrestlers = [Wrestler(env, name, i, pop, height, weight, exp) 
                     for i, (name, pop, height, weight, exp) in enumerate(wrestlers_data)]
    random.shuffle(env.wrestlers)  # Randomize entry order
    
    # Set normalization bounds
    heights = [w.height for w in env.wrestlers]
    weights = [w.weight for w in env.wrestlers]

    Wrestler.height_min = min(heights)
    Wrestler.height_max = max(heights)
    Wrestler.weight_min = min(weights)
    Wrestler.weight_max = max(weights)

    print("\nBattle Royale Entry Order:")
    for i, w in enumerate(env.wrestlers, 1):
        print(f"{i}. {w.name} - Initial Health: {w.health}")
    print("\n")
    env.reset()

    # Set all_wrestlers in viz.stats
    viz.stats["all_wrestlers"] = env.wrestlers
    
    # Create agents for each wrestler
    agents = {w.id: WrestlingAgent(w) for w in env.wrestlers}
    
    generation = 0
    while generation < num_generations:
        print(f"\n=== Generation {generation + 1} ===")
        env.reset()
        env._add_new_wrestler()
        env._add_new_wrestler()

        running = True
        clock = pygame.time.Clock()
        timestep_delay = 100  
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Get observations and actions
            obs = env._get_obs()
            actions = {w.id: agents[w.id].choose_action(obs.get(w.id, np.zeros(7))) for w in env.active_wrestlers}
            # Here we need to get intiator's action out of all these active_wrestler's actions
            obs, rewards, dones, infos, initiator, responder = env.step(actions)
            
            # Update visualization stats
            viz.stats["current_wrestlers"] = env.active_wrestlers
            viz.stats["eliminated"] = env.eliminated_wrestlers
            
            print(f"Rendering {len(env.active_wrestlers)} wrestlers")
            viz.render(env.active_wrestlers, initiator, responder)
            clock.tick(30)
            pygame.time.delay(timestep_delay)
            
            # Check for winner
            winner = next((w for w in env.active_wrestlers if infos.get(w.id, {}).get("winner", False)), None)
            if winner:
                viz.stats["winner"] = winner
                print(f"\nBattle Royale Winner: {winner.name}!")
                running = False
            
            # Pause briefly when match ends
            if any(dones.values()) and len(env.active_wrestlers) <= 1:
                pygame.time.delay(timestep_delay)

        # Evolve wrestlers for the next generation
        if generation < num_generations - 1:
            env.wrestlers = evolve_wrestlers(env, env.wrestlers, num_generations=1)
            viz.stats["all_wrestlers"] = env.wrestlers
            agents = {w.id: WrestlingAgent(w) for w in env.wrestlers}
        generation += 1 
    
    pygame.time.delay(5000)
    viz.close()
    sys.exit()

if __name__ == "__main__":
    run_battle_royale(num_generations=2)