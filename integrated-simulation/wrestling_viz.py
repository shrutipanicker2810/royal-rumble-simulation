import pygame
import numpy as np

class WrestlingViz:
    """Visualization class for the wrestling battle royale.
    
    Handles rendering of wrestlers, ring, and stats panel with scrolling capability.
    """
    
    def __init__(self, ring_size=4.0, screen_width=1000, screen_height=600):
        """Initialize visualization with screen dimensions and ring size."""
        pygame.init()
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Wrestling Battle Royale")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        self.title_font = pygame.font.SysFont("Arial", 24, bold=True)
        self.stats_font = pygame.font.SysFont("Arial", 15, bold=True)
        self.previous_health = {}
        self.responder_health_loss = 0

        # Color definitions
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.CRIMSON = (220, 20, 60)
        self.BLUE = (0, 0, 205)
        self.CYAN = (0,191,255)
        self.GREEN = (10, 168, 13)
        self.YELLOW = (255, 255, 0)
        self.SKIN_COLOR = (240, 164, 12)
        self.GREY = (40, 40, 50) #(150, 150, 150)
        
        # Ring dimensions and scaling
        self.ring_size = ring_size
        self.scale = min(screen_width * 0.9, screen_height) / (2 * ring_size * 1.1)
        self.ring_rect = pygame.Rect(
            (screen_width * 0.6 - self.ring_size * 2 * self.scale) / 2,
            (screen_height - self.ring_size * 2 * self.scale) / 2,
            self.ring_size * 2 * self.scale,
            self.ring_size * 2 * self.scale
        )
        
        # Stats tracking
        self.stats = {"current_wrestlers": [], "eliminated": [], "winner": None}
        self.initiator = None  # Currently attacking wrestler
        self.responder = None  # Currently defending wrestler
        
        # Panel dimensions
        self.panel_width = 410  # Width of stats panel
        self.panel_height = screen_height - 20  # Visible panel height
        self.column_width = (self.panel_width - 30) // 2  # Two columns with padding
        self.card_height = (self.panel_height - 40) // 5  # # 5 cards per column, accounting for title space

    def draw_humanoid(self, wrestler, screen_pos, is_left):
        """Draw a wrestler as a humanoid figure at the given screen position.
        
        Args:
            wrestler: The wrestler to draw
            screen_pos: (x,y) screen coordinates
            is_left: Whether this is the left-most wrestler (unused)
        """
        # Color wrestler based on role (attacker/defender/neutral)
        if wrestler == self.initiator:
            color = self.CRIMSON
        elif wrestler == self.responder:
            color = self.BLUE
        else:
            color = self.SKIN_COLOR
        
        # Scaling for wrestler size
        player_scale = 1.5
        head_radius = int(15 * player_scale)
        body_length = int(45 * player_scale)
        arm_length = int(20 * player_scale)
        leg_length = int(15 * player_scale)
        leg_pos = int(70 * player_scale)
        line_thickness = int(5 * player_scale)

        # Draw head and body
        pygame.draw.circle(self.screen, color, screen_pos, head_radius)        
        body_top = (screen_pos[0], screen_pos[1] + head_radius)
        body_bottom = (screen_pos[0], screen_pos[1] + body_length)
        pygame.draw.line(self.screen, color, body_top, body_bottom, line_thickness)
        
        # Handle attack animations
        action = wrestler.last_action
        action_time = wrestler.last_action_time
        if action in [0, 1, 3] and pygame.time.get_ticks() - action_time < 500:
            # Animate attacking limbs
            progress = min((pygame.time.get_ticks() - action_time) / 500, 1.0)
            self.draw_attack_limbs(wrestler, screen_pos, body_top, body_bottom, action, progress, color, player_scale)
        else:
            # Draw neutral stance
            pygame.draw.line(self.screen, color, body_top, (screen_pos[0] - arm_length, screen_pos[1] + arm_length), line_thickness)
            pygame.draw.line(self.screen, color, body_top, (screen_pos[0] + arm_length, screen_pos[1] + arm_length), line_thickness)
            pygame.draw.line(self.screen, color, body_bottom, (screen_pos[0] - leg_length, screen_pos[1] + leg_pos), line_thickness)
            pygame.draw.line(self.screen, color, body_bottom, (screen_pos[0] + leg_length, screen_pos[1] + leg_pos), line_thickness)
        
        # # Draw health bar above wrestler
        # health_ratio = wrestler.health / wrestler.max_health
        # bar_width = int(40 * player_scale)
        # bar_height = int(5 * player_scale)
        # bar_rect = (screen_pos[0] - bar_width // 2, screen_pos[1] - 40 * player_scale, bar_width, bar_height)
        # pygame.draw.rect(self.screen, self.BLACK, bar_rect, 2)
        # pygame.draw.rect(self.screen, (100, 100, 100), bar_rect)
        # pygame.draw.rect(self.screen, self.GREEN if health_ratio > 0.5 else self.YELLOW if health_ratio > 0.25 else self.RED,
        #                  (screen_pos[0] - bar_width // 2, screen_pos[1] - 40 * player_scale, int(bar_width * health_ratio), bar_height))
        
        # change tag background color for initiators and responders
        # Draw name tag
        name_font = pygame.font.SysFont("Arial", int(16 * player_scale))
        name_text = name_font.render(wrestler.name, True, self.WHITE)
        name_rect = name_text.get_rect(center=(screen_pos[0], screen_pos[1] - 40 * player_scale)) # change from -40 to -60, if health bar needs to be included
        background_rect = name_rect.inflate(10 * player_scale, 6 * player_scale)
        pygame.draw.rect(self.screen, (50, 50, 50, 200), background_rect, border_radius=5)
        pygame.draw.rect(self.screen, self.CRIMSON if wrestler == self.initiator else self.BLUE if wrestler == self.responder else self.WHITE, background_rect, 2, border_radius=5)
        self.screen.blit(name_text, name_rect)

    def draw_attack_limbs(self, wrestler, screen_pos, body_top, body_bottom, action, progress, color, player_scale):
        """Draw limbs in attacking positions based on action type.
        
        Args:
            wrestler: The attacking wrestler
            screen_pos: Position on screen
            body_top: Top of body line
            body_bottom: Bottom of body line
            action: Type of attack (0=punch, 1=kick, 3=signature)
            progress: Animation progress (0-1)
            color: Color to draw with
            player_scale: Size scaling factor
        """
        head_radius = int(15 * player_scale)
        arm_length = int(20 * player_scale)
        leg_length = int(15 * player_scale)
        leg_pos = int(70 * player_scale)
        line_thickness = int(5 * player_scale)
        punch_length = int(40 * player_scale)
        kick_length = int(30 * player_scale)

        if action == 0:  # Punch animation
            pygame.draw.line(self.screen, color, body_top, (screen_pos[0] + (punch_length if wrestler.id % 2 else -punch_length) * progress, screen_pos[1] + arm_length), line_thickness)
            pygame.draw.line(self.screen, color, body_top, (screen_pos[0] - (arm_length if wrestler.id % 2 else -arm_length), screen_pos[1] + arm_length), line_thickness)
            pygame.draw.line(self.screen, color, body_bottom, (screen_pos[0] - leg_length, screen_pos[1] + leg_pos), line_thickness)
            pygame.draw.line(self.screen, color, body_bottom, (screen_pos[0] + leg_length, screen_pos[1] + leg_pos), line_thickness)
        elif action == 1:  # Kick animation
            pygame.draw.line(self.screen, color, body_bottom, (screen_pos[0] + (kick_length if wrestler.id % 2 else -kick_length) * progress, screen_pos[1] + leg_pos + kick_length * progress), line_thickness)
            pygame.draw.line(self.screen, color, body_bottom, (screen_pos[0] - (leg_length if wrestler.id % 2 else -leg_length), screen_pos[1] + leg_pos), line_thickness)
            pygame.draw.line(self.screen, color, body_top, (screen_pos[0] - arm_length, screen_pos[1] + arm_length), line_thickness)
            pygame.draw.line(self.screen, color, body_top, (screen_pos[0] + arm_length, screen_pos[1] + arm_length), line_thickness)
        elif action == 3:  # Signature move animation
            pygame.draw.line(self.screen, color, body_top, (screen_pos[0] - arm_length, screen_pos[1] - arm_length * progress), line_thickness)
            pygame.draw.line(self.screen, color, body_top, (screen_pos[0] + arm_length, screen_pos[1] - arm_length * progress), line_thickness)
            pygame.draw.line(self.screen, color, body_bottom, (screen_pos[0] - leg_length, screen_pos[1] + leg_pos), line_thickness)
            pygame.draw.line(self.screen, color, body_bottom, (screen_pos[0] + leg_length, screen_pos[1] + leg_pos), line_thickness)
        
        # Draw hit effect if attack connects mid-animation
        if 0.4 < progress < 0.6 and pygame.time.get_ticks() - wrestler.last_hit_time < 100:
            pygame.draw.circle(self.screen, self.YELLOW, (int(screen_pos[0] + (punch_length if wrestler.id % 2 else -punch_length)), int(screen_pos[1] + 30 * player_scale)), 
                               int(15 * player_scale * (1 - abs(progress - 0.5) * 2)), 2)

    def pos_to_screen(self, pos, is_left):
        """Convert world coordinates to screen coordinates."""
        x = self.ring_rect.centerx + (pos[0] * self.scale)
        y = self.ring_rect.centery - (pos[1] * self.scale)
        return (int(x), int(y))

    def draw_ring(self):
        """Draw the wrestling ring and ropes."""
        self.screen.fill((50, 50, 70))  # Dark blue background
        pygame.draw.rect(self.screen, (240, 220, 180), self.ring_rect)  # Light tan ring canvas
        rope_colors = [(200, 200, 200), (180, 180, 180), (160, 160, 160)]  # Top to bottom rope colors
        for i in range(3):  # Draw three ropes
            rope_y = self.ring_rect.top + (i+1)*self.ring_rect.height//4
            pygame.draw.line(self.screen, rope_colors[i], (self.ring_rect.left, rope_y), (self.ring_rect.right, rope_y), 3)

    def draw_stats_panel(self):
        """Draw the stats panel with fixed cards for all 10 wrestlers."""
        # Calculate total content height
        panel = pygame.Surface((self.panel_width, self.panel_height))
        panel.fill((40, 40, 50))  # Dark background
        
        # Draw title
        title = self.title_font.render("Current Wrestlers", True, self.WHITE)
        panel.blit(title, (10, 10))
        y_pos = 40
        
        # Ensure we have a list of all wrestlers (set in render)
        all_wrestlers = self.stats["all_wrestlers"]
        active_wrestlers = self.stats["current_wrestlers"]
        eliminated_wrestlers = self.stats["eliminated"]
        
        for col in range(2):  # Two columns
            x_offset = 10 if col == 0 else self.column_width + 20  # Left or right column
            column_wrestlers = all_wrestlers[col * 5:(col + 1) * 5]  # First 5 or last 5
            y_pos = 40  # Reset y_pos for each column

            for i, wrestler in enumerate(column_wrestlers):
                is_active = wrestler in active_wrestlers
                is_eliminated = wrestler in eliminated_wrestlers
                is_initiator = wrestler == self.initiator
                is_responder = wrestler == self.responder
                
                # Determine card color and text color
                if is_initiator:
                    card_color = self.CRIMSON
                    text_color = self.WHITE
                elif is_responder:
                    card_color = self.BLUE
                    text_color = self.WHITE
                elif is_active:
                    card_color = (60, 60, 70)  # Neutral active color
                    text_color = self.WHITE
                else:
                    card_color = (40, 40, 50)
                    text_color = self.BLACK  # Inactive (not yet entered or eliminated)
                
                # Draw card background
                card_rect = (x_offset, y_pos, self.column_width - 5, self.card_height - 10)
                pygame.draw.rect(panel, card_color, card_rect)
                
                if wrestler:
                    # Draw name
                    name = self.title_font.render(wrestler.name, True, self.BLACK)
                    panel.blit(name, (x_offset + 7, y_pos + 7))
                    name = self.title_font.render(wrestler.name, True, text_color)
                    panel.blit(name, (x_offset + 5, y_pos + 5))
                    
                    # "HP" label before health bar ---
                    hp_label = self.stats_font.render("HP", True, text_color)
                    panel.blit(hp_label, (x_offset + 5, y_pos + 32))
                    
                    # Health bar and value (shifted right to accommodate "HP" label)
                    health_ratio = wrestler.health / wrestler.max_health if is_active or is_eliminated else 1.0
                    health_value = wrestler.health if is_active or is_eliminated else wrestler.max_health
                    health_bar_width = (self.column_width - 70)  
                    health_bar_rect = (x_offset + 25, y_pos + 35, health_bar_width, 10)  
                    pygame.draw.rect(panel, self.BLACK, health_bar_rect, 3)
                    pygame.draw.rect(panel, (100, 100, 100), health_bar_rect)
                    pygame.draw.rect(panel, self.GREEN if health_ratio > 0.5 else self.YELLOW if health_ratio > 0.25 else self.RED,
                                    (x_offset + 25, y_pos + 35, int(health_bar_width * health_ratio), 10))
                    health_text = self.stats_font.render(f"{health_value:.1f}", True, text_color)
                    panel.blit(health_text, (x_offset + 25 + health_bar_width + 5, y_pos + 32))  
                    
                    # "ST" label before stamina bar ---
                    sta_label = self.stats_font.render("ST", True, text_color)
                    panel.blit(sta_label, (x_offset + 5, y_pos + 52))
                    
                    # Stamina bar and value (shifted right to accommodate "STA" label)
                    stamina_ratio = wrestler.stamina / wrestler.max_stamina if is_active else 1.0
                    stamina_value = wrestler.stamina if is_active else wrestler.max_stamina
                    stamina_bar_rect = (x_offset + 25, y_pos + 55, health_bar_width, 10)  
                    pygame.draw.rect(panel, self.BLACK, stamina_bar_rect, 3)
                    pygame.draw.rect(panel, (100, 100, 100), stamina_bar_rect)
                    pygame.draw.rect(panel, self.CYAN, (x_offset + 25, y_pos + 55, int(health_bar_width * stamina_ratio), 10))
                    stamina_text = self.stats_font.render(f"{stamina_value:.0f}", True, text_color)
                    panel.blit(stamina_text, (x_offset + 25 + health_bar_width + 5, y_pos + 52))  

                    last_action = wrestler.last_action
                    action_type = None
                    if last_action == 0:
                        action_type = "Punch"
                    if last_action == 1:
                        action_type = "Kick"
                    if last_action == 3:
                        action_type = "Signature"
                    if last_action == 4:
                        action_type = "No-op"
                    move_label = self.stats_font.render("MOVE = " if wrestler == self.initiator else "", True, text_color)
                    panel.blit(move_label, (x_offset + 5, y_pos + 72))
                    action_text = self.stats_font.render(f"{action_type}" if wrestler == self.initiator else "", True, text_color)
                    panel.blit(action_text, (x_offset + 55, y_pos + 72))  

                    # Draw health loss for the responder
                    if wrestler == self.responder and self.responder_health_loss > 0:
                        health_loss_text = self.stats_font.render(f"HP LOST: {self.responder_health_loss:.1f}", True, text_color)
                        panel.blit(health_loss_text, (x_offset + 5, y_pos + 72))
                    
                    if wrestler == self.initiator and self.responder:
                        # Calculate the distance between initiator and responder
                        initiator_pos = self.initiator.get_qpos()
                        responder_pos = self.responder.get_qpos()
                        distance = np.linalg.norm(initiator_pos - responder_pos)
                        distance_text = self.stats_font.render(f"DIST: {distance:.1f}", True, text_color)
                        panel.blit(distance_text, (x_offset + health_bar_width + 5, y_pos + 72))  
                y_pos += self.card_height
        
        # Blit the panel onto the screen
        panel_start_x = int(self.ring_rect.right + 10)  # Start 10 pixels after the ring's right edge
        self.screen.blit(panel, (panel_start_x, 10))
        
       
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True
    

    def render(self, wrestlers, initiator=None, responder=None):
        """Render the current state of the match.
        
        Args:
            wrestlers: List of active wrestlers
            initiator: Wrestler initiating action (None if none)
            responder: Wrestler responding to action (None if none)
            
        Returns:
            bool: False if user requested quit, True otherwise
        """
        self.responder_health_loss = 0
        if responder:
            responder_id = id(responder)
            current_health = responder.health
            previous_health = self.previous_health.get(responder_id, current_health)
            self.responder_health_loss = max(0, previous_health - current_health)
            self.previous_health[responder_id] = current_health

        # Update stats and handle events
        self.initiator = initiator
        self.responder = responder
        self.stats["current_wrestlers"] = wrestlers
        if not self.stats["all_wrestlers"]:
            self.stats["all_wrestlers"] = wrestlers[:10]  # Limit to 10 for display
        if not self.handle_events():
            return False
        
        self.screen.fill((50, 50, 70))  # Dark blue background
        self.draw_ring()
        # Draw the title at the top center
        title_font = pygame.font.SysFont("Arial", 32, bold=True)
        title_text = title_font.render("ROYAL RUMBLE WRESTLING SIMULATION", True, self.WHITE)
        title_rect = title_text.get_rect(center=(self.ring_rect.centerx, 30))
        background_rect = title_rect.inflate(20, 10)  
        pygame.draw.rect(self.screen, self.BLACK, background_rect)  # Black background
        pygame.draw.rect(self.screen, self.WHITE, background_rect, 2)  # White border
        self.screen.blit(title_text, title_rect)

        # Draw all components        
        for i, wrestler in enumerate(wrestlers):
            screen_pos = self.pos_to_screen(wrestler.get_qpos(), i == 0)
            self.draw_humanoid(wrestler, screen_pos, i == 0)
        self.draw_stats_panel()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(30)
        return True

    def close(self):
        """Clean up pygame resources."""
        pygame.quit()