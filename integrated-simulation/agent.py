import numpy as np

class WrestlingAgent:
    """Agent that controls a wrestler's actions based on game state."""
    
    def __init__(self, wrestler):
        """Initialize with the wrestler this agent controls."""
        self.wrestler = wrestler

    def choose_action(self, observation):
        """Choose an action based on current observation.
        
        Args:
            observation: Current state observation
            
        Returns:
            int: Action index (0-4)
        """
        if self.wrestler.stamina <= 0:
            return 4  # No-op if out of stamina

        # Calculate distance to opponent
        distance = np.linalg.norm(observation[3:5] - observation[5:7]) if len(observation) >= 7 else 2.0
        strength, agility, defensiveness = self.wrestler.genes

        action_probs = np.zeros(5)  # Initialize action probabilities
        
        if distance > 1.5:  # Far from opponent - recover stamina
            action_probs[4] = 0.8  # No probability for no-op
            action_probs[0] = 0.1  # Small chance to punch
            action_probs[1] = 0.1  # Small chance to kick
        else:  # Close to opponent - attack
            # Weight actions by wrestler's strengths
            action_probs[0] = strength * 0.5  # Punch favors strength
            action_probs[1] = agility * 0.4    # Kick favors agility
            action_probs[3] = strength * 0.3   # Signature move favors strength
            action_probs[4] = 0              # Small chance to do nothing

        # Normalize probabilities and choose action
        action_probs /= action_probs.sum()
        return np.random.choice(5, p=action_probs)