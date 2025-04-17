import numpy as np
import random
from wrestler import Wrestler  

class GeneticEvolution:
    def __init__(self, env, population_size=10, num_generations=3, mutation_rate=0.1):
        self.env = env
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.population = None
        self.fitness_history = []
        self.wrestler_history = {}
        self.lineage_map = {}  # Track parent-to-child relationships
    
    def initialize_population(self, initial_wrestlers):
        self.population = initial_wrestlers[:self.population_size]
        if len(self.population) < self.population_size:
            for i in range(len(self.population), self.population_size):
                base = random.choice(initial_wrestlers)
                new_wrestler = Wrestler(self.env, f"Random-{i}", i, 
                                      base.popularity, base.height, base.weight, base.experience)
                new_wrestler.genes = np.random.uniform(0, 1, 3)
                self.population.append(new_wrestler)
        print(f"Initialized population with {len(self.population)} wrestlers.")
        self.print_population_details("Initial Population")
    
    def evaluate_fitness(self):
        fitness_scores = []
        for wrestler in self.population:
            health_factor = wrestler.health / wrestler.max_health
            stamina_factor = wrestler.stamina / wrestler.max_stamina
            damage_factor = (wrestler.max_health - wrestler.health) / 100
            win_factor = 10.0 if wrestler.health > 0 and len(self.env.active_wrestlers) == 1 else 0.0
            
            fitness = (0.3 * health_factor + 
                      0.2 * stamina_factor + 
                      0.3 * damage_factor + 
                      0.2 * win_factor)
            fitness_scores.append((wrestler, fitness))
            if wrestler.id not in self.wrestler_history:
                self.wrestler_history[wrestler.id] = []
            self.wrestler_history[wrestler.id].append(fitness)
        return fitness_scores
    
    def select_parents(self, fitness_scores):
        avg_fitness = np.mean([f for _, f in fitness_scores])
        below_average = [(w, f) for w, f in fitness_scores if f < avg_fitness]
        if not below_average:
            below_average = fitness_scores
        
        num_parents = min(self.population_size // 2, len(below_average))
        parents = []
        
        print(f"\nSelected Parents (Below Average Fitness, Avg: {avg_fitness:.2f}):")
        for _ in range(num_parents):
            wrestler, fitness = random.choice(below_average)
            parents.append(wrestler)
            print(f"  {wrestler.name} (Fitness: {fitness:.2f}, Genes: {wrestler.genes.round(2)})")
            below_average.remove((wrestler, fitness))
        return parents
    
    def crossover(self, parent1, parent2, child_id):
        child_genes = np.zeros(3)
        for i in range(3):
            child_genes[i] = parent1.genes[i] if random.random() < 0.5 else parent2.genes[i]
        
        child_name = f"{parent1.name}-E-{child_id}"
        child = Wrestler(self.env, child_name, child_id,
                        (parent1.popularity + parent2.popularity) / 2,
                        (parent1.height + parent2.height) / 2,
                        (parent1.weight + parent2.weight) / 2,
                        (parent1.experience + parent2.experience) / 2)
        child.genes = child_genes
        print(f"  Crossover: {parent1.name} + {parent2.name} -> {child.name} (Genes: {child_genes.round(2)})")
        self.lineage_map[child.id] = parent1.id  # Map child to parent1
        return child
    
    def mutate(self, wrestler):
        original_genes = wrestler.genes.copy()
        for i in range(3):
            if random.random() < self.mutation_rate:
                wrestler.genes[i] = np.clip(wrestler.genes[i] + random.uniform(-0.2, 0.2), 0, 1)
        if not np.array_equal(original_genes, wrestler.genes):
            print(f"  Mutation: {wrestler.name} (Genes: {original_genes.round(2)} -> {wrestler.genes.round(2)})")
    
    def replace_population(self, fitness_scores, offspring):
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        new_population = [w for w, _ in fitness_scores]
        
        for i, (parent, _) in enumerate(fitness_scores):
            for child in offspring:
                if child.name.startswith(parent.name.split("-E-")[0] + "-E-"):
                    child.health = child.max_health
                    child.stamina = child.max_stamina
                    new_population[i] = child
                    break
        return new_population[:self.population_size]
    
    def evolve(self, initial_wrestlers):
        self.initialize_population(initial_wrestlers)
        
        for generation in range(self.num_generations):
            print(f"\n=== Evolving Generation {generation + 1}/{self.num_generations} ===")
            
            fitness_scores = self.evaluate_fitness()
            avg_fitness = np.mean([f for _, f in fitness_scores])
            max_fitness = max([f for _, f in fitness_scores])
            self.fitness_history.append((generation + 1, avg_fitness, max_fitness))
            print(f"Average Fitness: {avg_fitness:.2f}, Max Fitness: {max_fitness:.2f}")
            self.print_population_details(f"Generation {generation + 1} Before Evolution")
            
            parents = self.select_parents(fitness_scores)
            
            print("\nCrossover and Mutation Events:")
            offspring = []
            child_id_counter = len(self.population)
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1 = self.crossover(parents[i], parents[i + 1], child_id_counter)
                    child_id_counter += 1
                    child2 = self.crossover(parents[i + 1], parents[i], child_id_counter)
                    child_id_counter += 1
                    self.mutate(child1)
                    self.mutate(child2)
                    offspring.extend([child1, child2])
            
            self.population = self.replace_population(fitness_scores, offspring)
            for wrestler in self.population:
                if "-E-" in wrestler.name and wrestler.id >= child_id_counter - len(offspring):
                    wrestler.health = wrestler.max_health
                    wrestler.stamina = wrestler.max_stamina
            fitness_scores = self.evaluate_fitness()
            avg_fitness = np.mean([f for _, f in fitness_scores])
            max_fitness = max([f for _, f in fitness_scores])
            self.fitness_history.append((generation + 1.5, avg_fitness, max_fitness))
            self.print_population_details(f"Generation {generation + 1} After Evolution")
            
            for wrestler in self.population:
                wrestler.health = wrestler.max_health
                wrestler.stamina = wrestler.max_stamina
                wrestler.match_pos = None
                wrestler._opponents = []
        
        self.print_performance_summary()
        self.print_wrestler_evolution_summary()
        print("Evolution complete!")
        return self.population
    
    def print_population_details(self, title):
        print(f"\n{title}:")
        print("ID  | Name                  | Fitness | Genes [Str, Agi, Def]")
        print("----|-----------------------|---------|----------------------")
        for wrestler, fitness in sorted(self.evaluate_fitness(), key=lambda x: x[1], reverse=True):
            genes_str = f"[{wrestler.genes[0]:.2f}, {wrestler.genes[1]:.2f}, {wrestler.genes[2]:.2f}]"
            print(f"{wrestler.id:3d} | {wrestler.name:<21} | {fitness:7.2f} | {genes_str}")
    
    def print_performance_summary(self):
        print("\n=== Population Performance Summary ===")
        print("Generation | Avg Fitness | Max Fitness")
        print("-----------|-------------|------------")
        for gen, avg, max_f in self.fitness_history:
            print(f"{gen:10.1f} | {avg:11.2f} | {max_f:11.2f}")
        
        initial_avg = self.fitness_history[0][1]
        final_avg = self.fitness_history[-1][1]
        initial_max = self.fitness_history[0][2]
        final_max = self.fitness_history[-1][2]
        
        avg_improvement = ((final_avg - initial_avg) / initial_avg * 100) if initial_avg > 0 else 0
        max_improvement = ((final_max - initial_max) / initial_max * 100) if initial_max > 0 else 0
        
        print(f"\nImprovement Over {self.num_generations} Generations:")
        print(f"Average Fitness: {initial_avg:.2f} -> {final_avg:.2f} ({avg_improvement:+.2f}%)")
        print(f"Maximum Fitness: {initial_max:.2f} -> {final_max:.2f} ({max_improvement:+.2f}%)")
    
    def print_wrestler_evolution_summary(self):
        print("\n=== Individual Wrestler Evolution Summary ===")
        print("ID  | Name                  | Initial Fitness | Final Fitness | Change (%)")
        print("----|-----------------------|-----------------|---------------|-----------")
        for wrestler in self.population:
            wrestler_id = wrestler.id
            # Check if this wrestler evolved from another
            original_id = self.lineage_map.get(wrestler_id, wrestler_id)
            fitness_list = self.wrestler_history.get(original_id, []) + self.wrestler_history.get(wrestler_id, [])
            if len(fitness_list) >= 1:  # Show all wrestlers, even with one entry
                initial_fitness = fitness_list[0]
                final_fitness = fitness_list[-1]
                change = ((final_fitness - initial_fitness) / initial_fitness * 100) if initial_fitness > 0 else 0
                print(f"{wrestler.id:3d} | {wrestler.name:<21} | {initial_fitness:15.2f} | {final_fitness:13.2f} | {change:9.2f}")

def evolve_wrestlers(env, initial_wrestlers, num_generations=3):
    ge = GeneticEvolution(env, population_size=len(initial_wrestlers), num_generations=num_generations)
    evolved_population = ge.evolve(initial_wrestlers)
    env.wrestlers = evolved_population
    env.reset()
    return evolved_population