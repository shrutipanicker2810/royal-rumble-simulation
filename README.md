########### collaborations folder is where all four of us have worked to create backend and experiment on simulations.
->PG_impl.py is the Policy Gradient experiment file.
→ It uses movements.py and variables.py as the backend to define the wrestling environment and game dynamics.
    1. The initiator is selected using Simulated Annealing.
    2. A stochastic policy network (built using Keras) chooses an action: punch, kick, signature 
        move, or stay idle.
    3. If the responder is eliminated after an action (reward[initiator] += 100), a new wrestler 
        enters the ring.
    4. The environment handles stamina regeneration for idle moves and penalizes low stamina 
         attacks.
    5. Rewards are calculated after each action using damage dealt and stamina consumed.
    
→ The WrestlingEnv is used to simulate full rounds, collect trajectories, compute discounted returns, and update the policy using the REINFORCE algorithm.

########### main folder includes the code where we later on finalized the simulation and integrated our backend code with.
-> wrestling_battle_royale.py is the main file 
    1. BattleRoyaleEnv.__init__(ring_size, entry_interval) - Sets up ring dimensions, entry timing, wrestler lists, positions, and simulated‑annealing parameters.
    2. BattleRoyaleEnv.reset() - Clears current match state (timers, active/eliminated lists, positions) and returns the initial observations.
    3. BattleRoyaleEnv.step(actions) - Advances one timestep:
        3.1 Checks whether to spawn a new wrestler
        3.2 Selects an initiator and responder (SA + random)
        3.3 Applies the initiator’s action (movement, damage calculation, stamina)
        3.4 Handles eliminations, reward bookkeeping, “done” flags, and returns (obs, rewards, dones, infos, initiator, responder).
    4. BattleRoyaleEnv._add_new_wrestler() - Picks the next available wrestler, finds a random valid spot (respecting MIN_SEPARATION), and updates opponent lists.
    5. BattleRoyaleEnv.calculate_fitness(wrestler) - Computes a weighted score of health, stamina, experience, and popularity for SA.
    6. BattleRoyaleEnv._select_combatants() - Runs a few iterations of simulated annealing to choose an “initiator,” then picks a random “responder.”
    7. BattleRoyaleEnv._check_hit(attacker, defender) - Returns True if the attacker and defender are within hit‑range (Euclidean distance < 1.5).
    8. BattleRoyaleEnv._handle_elimination(wrestler) - Moves a fallen wrestler to the eliminated list and removes them from all opponents’ lists.
    9. BattleRoyaleEnv._update_simulation() - Applies per‑wrestler movement (initial center pull, repulsion forces, and boundary clipping).
    10. BattleRoyaleEnv._get_obs() - Builds and returns a dict of observation vectors ([health, stamina, last_action, self_pos, nearest_opp_pos]) keyed by wrestler ID.
    11. run_battle_royale(num_generations) (MAIN function) - Initializes Pygame, env, viz, and agents
        11.1 Loads/shuffles the wrestler roster and sets normalization bounds
        11.2 Loops through generations, running full matches, rendering each frame
        11.3 Optionally calls evolve_wrestlers(...) between generations

-> wrestler.py
class Wrestler - Encapsulates a single wrestler’s stats and state. Key methods typically include:
    1. __init__(env, name, id, popularity, height, weight, experience)
    2. compute_strength() / compute_stamina() / compute_defense_rating()
    3. apply_action(action)
    4. set_match_position(pos_idx)
    5. set_opponents(opponent_list)
    6. get_qpos() (returns its current 2D position vector for observations)

-> agent.py
class WrestlingAgent - Wraps a policy or heuristic for choosing actions. Includes:
    1. __init__(wrestler)
    2. choose_action(observation) → returns an integer in {0: punch, 1: kick, 3: signature, 2: no‑op}

-> wrestling_viz.py
class WrestlingViz - Handles all Pygame rendering and on‑screen stats. Includes:
    1. __init__() - render(active_wrestlers, initiator, responder) (draw ring, wrestlers, highlights)
    2. close() (cleanly shut down Pygame) 
    
-> genetic_evolution.py
    1. evolve_wrestlers(env, wrestlers, num_generations) - Applies the genetic‑algorithm routines to the current roster—mutating and selecting new variants—and returns an updated list of Wrestler instances for the next generation.
