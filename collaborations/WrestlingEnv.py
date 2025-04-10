import pygame
import sys
import random
import math


SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
FPS = 30
MIN_DISTANCE = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLORS = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0),
          (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
          (128, 128, 0), (128, 0, 128)]


class Wrestler:
    def __init__(self, wrestler_id, name):
        self.id = wrestler_id
        self.name = name
        self.color = COLORS[wrestler_id % len(COLORS)]
        self.health = 200
        self.stamina = 100
        self.speed = 3
        self.attack_range = 60
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.action = "Idle"
        self.action_in_progress = False
        self.action_timer = 0
        self.target = None
        self.eliminated = False
        self.attack_type = None

    def move_toward(self, target_x, target_y):
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.hypot(dx, dy)
        if distance > 0:
            dx = (dx / distance) * self.speed
            dy = (dy / distance) * self.speed
        self.x += dx
        self.y += dy

    def check_collision(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        return math.hypot(dx, dy) < MIN_DISTANCE

    def resolve_collision(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        distance = math.hypot(dx, dy)

        if distance == 0:
            return

        overlap = (MIN_DISTANCE - distance) / 2
        ratio = overlap / distance

        self.x += dx * ratio
        self.y += dy * ratio
        other.x -= dx * ratio
        other.y -= dy * ratio

    def update(self, wrestlers):
        if self.health <= 0:
            self.eliminated = True
            return

        if not self.action_in_progress:
            opponents = [w for w in wrestlers if w != self and not w.eliminated]
            if opponents:
                self.target = min(opponents, key=lambda w: math.hypot(w.x - self.x, w.y - self.y))
                self.move_toward(self.target.x, self.target.y)

                distance = math.hypot(self.x - self.target.x, self.y - self.target.y)
                if distance < self.attack_range and random.random() < 0.3:
                    self.perform_attack()

    def perform_attack(self):
        if self.action_in_progress or not self.target:
            return

        self.attack_type = random.choices(
            ["Punch", "Kick", "Special"],
            weights=[0.5, 0.3, 0.2],
            k=1
        )[0]

        self.action = self.attack_type
        self.action_in_progress = True
        self.action_timer = pygame.time.get_ticks()

        if self.target:
            damage = random.randint(10, 20) if self.attack_type != "Special" else random.randint(25, 35)
            self.target.health = max(0, self.target.health - damage)
            self.stamina = max(0, self.stamina - 10)


class WrestlingEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)

        self.wrestlers = [
            Wrestler(0, "John Cena"),
            Wrestler(1, "The Rock"),
            Wrestler(2, "Undertaker"),
            Wrestler(3, "Triple H"),
            Wrestler(4, "Randy Orton"),
            Wrestler(5, "Brock Lesnar")
        ]

        self.active_wrestlers = []
        self.eliminated = []
        self.next_entry_time = pygame.time.get_ticks() + 5000
        self.match_over = False
        self.initial_positions_set = False

    def reset(self):
        # Set initial positions for first two wrestlers
        self.wrestlers[0].x = 150
        self.wrestlers[0].y = SCREEN_HEIGHT // 2
        self.wrestlers[1].x = SCREEN_WIDTH - 150
        self.wrestlers[1].y = SCREEN_HEIGHT // 2

        for w in self.wrestlers:
            w.health = 200
            w.stamina = 100
            w.eliminated = False

        self.active_wrestlers = self.wrestlers[:2]
        self.eliminated = []
        self.next_entry_time = pygame.time.get_ticks() + 5000
        self.match_over = False
        self.initial_positions_set = True

    def step(self):
        current_time = pygame.time.get_ticks()

        if not self.initial_positions_set:
            self.reset()

        # Handle new entries
        if (current_time > self.next_entry_time and
                len(self.active_wrestlers) < 6 and
                not self.match_over):

            eligible = [w for w in self.wrestlers
                        if w not in self.active_wrestlers
                        and not w.eliminated]

            if eligible:
                new_wrestler = random.choice(eligible)
                valid = False
                attempts = 0
                while not valid and attempts < 100:
                    new_x = random.randint(100, SCREEN_WIDTH - 100)
                    new_y = random.randint(100, SCREEN_HEIGHT - 100)
                    valid = True
                    for w in self.active_wrestlers:
                        if math.hypot(new_x - w.x, new_y - w.y) < MIN_DISTANCE:
                            valid = False
                            break
                    attempts += 1

                new_wrestler.x = new_x
                new_wrestler.y = new_y
                self.active_wrestlers.append(new_wrestler)
                self.next_entry_time = current_time + 5000 if len(eligible) > 1 else float('inf')

        # Update wrestlers
        active = [w for w in self.active_wrestlers if not w.eliminated]
        for wrestler in active:
            wrestler.update(active)

        # Handle collisions
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                if active[i].check_collision(active[j]):
                    active[i].resolve_collision(active[j])

        # Check eliminations
        for wrestler in active.copy():
            if wrestler.health <= 0:
                self.eliminated.append(wrestler)
                self.active_wrestlers.remove(wrestler)
                print(f"{wrestler.name} eliminated!")

        # Check match end conditions
        active_count = len([w for w in self.active_wrestlers if not w.eliminated])
        remaining = [w for w in self.wrestlers if w not in self.active_wrestlers and not w.eliminated]

        # End match if only 0 or 1 wrestlers left and no more coming
        if (active_count <= 1) and not remaining:
            self.match_over = True

    def draw_wrestler(self, wrestler):
        if wrestler.action_in_progress:
            self.animate_attack(wrestler)

        # Head
        pygame.draw.circle(self.screen, wrestler.color,
                           (int(wrestler.x), int(wrestler.y - 40)), 15)

        # Body
        body_top = (wrestler.x, wrestler.y - 30)
        body_bottom = (wrestler.x, wrestler.y)
        pygame.draw.line(self.screen, wrestler.color, body_top, body_bottom, 5)

        # Arms
        if wrestler.attack_type == "Punch":
            self.draw_punch_arms(wrestler, body_top)
        elif wrestler.attack_type == "Kick":
            self.draw_kick_legs(wrestler, body_bottom)
        else:
            self.draw_normal_arms(wrestler, body_top)
            self.draw_normal_legs(wrestler, body_bottom)

        # Health bar
        pygame.draw.rect(self.screen, BLACK, (wrestler.x - 35, wrestler.y - 70, 70, 12))
        pygame.draw.rect(self.screen, (255, 40, 40),
                         (wrestler.x - 33, wrestler.y - 68, (66 * wrestler.health) // 200, 8))

        # Name
        text = self.font.render(wrestler.name, True, BLACK)
        self.screen.blit(text, (wrestler.x - 40, wrestler.y - 90))

    def draw_punch_arms(self, wrestler, body_top):
        progress = self.get_attack_progress(wrestler)
        punch_length = 50 * progress

        pygame.draw.line(self.screen, wrestler.color,
                         body_top,
                         (wrestler.x + punch_length, wrestler.y - 10), 5)
        pygame.draw.line(self.screen, wrestler.color,
                         body_top,
                         (wrestler.x - 30, wrestler.y - 10), 5)

    def draw_kick_legs(self, wrestler, body_bottom):
        progress = self.get_attack_progress(wrestler)
        kick_height = 60 * progress

        pygame.draw.line(self.screen, wrestler.color,
                         body_bottom,
                         (wrestler.x + 40, wrestler.y + kick_height), 5)
        pygame.draw.line(self.screen, wrestler.color,
                         body_bottom,
                         (wrestler.x - 15, wrestler.y + 30), 5)

    def draw_normal_arms(self, wrestler, body_top):
        pygame.draw.line(self.screen, wrestler.color,
                         body_top,
                         (wrestler.x - 30, wrestler.y - 10), 5)
        pygame.draw.line(self.screen, wrestler.color,
                         body_top,
                         (wrestler.x + 30, wrestler.y - 10), 5)

    def draw_normal_legs(self, wrestler, body_bottom):
        pygame.draw.line(self.screen, wrestler.color,
                         body_bottom,
                         (wrestler.x - 15, wrestler.y + 40), 5)
        pygame.draw.line(self.screen, wrestler.color,
                         body_bottom,
                         (wrestler.x + 15, wrestler.y + 40), 5)

    def get_attack_progress(self, wrestler):
        elapsed = pygame.time.get_ticks() - wrestler.action_timer
        duration = 400
        return min(elapsed / duration, 1.0)

    def animate_attack(self, wrestler):
        progress = self.get_attack_progress(wrestler)

        if progress >= 1.0:
            wrestler.action_in_progress = False
            wrestler.action = "Idle"
            wrestler.attack_type = None
            return

        if 0.4 < progress < 0.6 and wrestler.target:
            pygame.draw.circle(self.screen, (255, 255, 0),
                               (int(wrestler.target.x), int(wrestler.target.y)),
                               int(20 * random.random()), 2)

    def render(self):
        self.screen.fill(WHITE)
        pygame.draw.rect(self.screen, BLACK, (50, 50, SCREEN_WIDTH - 100, SCREEN_HEIGHT - 100), 3)

        for wrestler in self.active_wrestlers:
            if not wrestler.eliminated:
                self.draw_wrestler(wrestler)


        text = self.font.render(f"Next entry: {max(0, (self.next_entry_time - pygame.time.get_ticks()) // 1000)}s",
                                True, BLACK)
        self.screen.blit(text, (10, 10))
        text = self.font.render(f"Eliminated: {len(self.eliminated)}", True, BLACK)
        self.screen.blit(text, (SCREEN_WIDTH - 200, 10))

        pygame.display.flip()
        self.clock.tick(FPS)




def main():

    env = WrestlingEnv()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if not env.match_over:
            env.step()
        env.render()

        if env.match_over:

            remaining = [w for w in env.active_wrestlers if not w.eliminated]

            if remaining:
                winner = remaining[0].name
                text = env.font.render(f"ROYAL RUMBLE WINNER: {winner}!", True, BLACK)
            else:
                text = env.font.render("NO SURVIVORS!", True, BLACK)

            env.screen.blit(text, (SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2))
            pygame.display.flip()


            pygame.time.wait(3000)
            pygame.quit()
            sys.exit()


if __name__ == "__main__":
    main()


