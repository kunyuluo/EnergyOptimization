import random
import matplotlib.pyplot as plt


# Define the objective function
def objective_function(x1, x2, x3, x4):
    return 118.1459 + 14.8214 * x1 + 28.4301 * x2 + 35.9796 * x3 - 8.3814 * x4


# Define constraints
def constraint1(x1):
    return 0.1 <= x1 <= 0.5


def constraint2(x2):
    return 0.3 <= x2 <= 0.8


def constraint3(x3):
    return 0.6 <= x3 <= 1.3


def constraint4(x4):
    return 3.0 <= x4 <= 7.0


def genetic_algorithm(population_size, num_generation):
    # Initialization
    population = []
    fitness_history = []

    for i in range(population_size):
        x1 = random.uniform(0.1, 0.5)
        x2 = random.uniform(0.2, 0.8)
        x3 = random.uniform(0.5, 1.3)
        x4 = random.uniform(3.0, 7.0)
        population.append((x1, x2, x3, x4))

    best_solution = None
    best_fitness = float('-inf')

    for generation in range(num_generation):
        # Evaluation
        fitness = [objective_function(x1, x2, x3, x4) for x1, x2, x3, x4 in population]

        feasible_population = [individual for individual in population
                               if constraint1(individual[0]) and constraint2(individual[1])
                               and constraint1(individual[2]) and constraint2(individual[3])]

        # Selection
        if feasible_population:
            feasible_fitness = [objective_function(x1, x2, x3, x4) for x1, x2, x3, x4 in feasible_population]
            parents = random.choices(feasible_population, weights=feasible_fitness, k=population_size)
        else:
            parents = []
            while len(parents) < population_size:
                potential_parents = random.choices(population, weights=fitness)[0]
                if constraint1(potential_parents[0]) and constraint2(potential_parents[1])\
                        and constraint3(potential_parents[2]) and constraint4(potential_parents[3]):
                    parents.append(potential_parents)

        # Crossover
        offspring = []
        for i in range(population_size):
            parent1, parent2, parent3, parent4 = random.choices(parents, k=4)
            x1_child = random.uniform(min(parent1[0], parent2[0], parent3[0], parent4[0]),
                                      max(parent1[0], parent2[0], parent3[0], parent4[0]))
            x2_child = random.uniform(min(parent1[1], parent2[1], parent3[1], parent4[1]),
                                      max(parent1[1], parent2[1], parent3[1], parent4[1]))
            x3_child = random.uniform(min(parent1[2], parent2[2], parent3[2], parent4[2]),
                                      max(parent1[2], parent2[2], parent3[2], parent4[2]))
            x4_child = random.uniform(min(parent1[3], parent2[3], parent3[3], parent4[3]),
                                      max(parent1[3], parent2[3], parent3[3], parent4[3]))
            offspring.append((x1_child, x2_child, x3_child, x4_child))

        # Mutation
        mutation_rate = 1 / (generation + 1)  # Dynamic mutation rate
        for i in range(population_size):
            if random.random() < mutation_rate:
                offspring[i] = (random.uniform(0.1, 0.5), random.uniform(0.2, 0.8),
                                random.uniform(0.5, 1.3), random.uniform(3.0, 7.0))

        # Elitism
        if best_solution is not None:
            offspring[0] = best_solution

        population = offspring

        # Find the best feasible solution
        feasible_solution = [(x1, x2, x3, x4) for (x1, x2, x3, x4) in population
                             if constraint1(x1) and constraint2(x2) and constraint3(x3) and constraint4(x4)]
        if feasible_solution:
            best_solution = min(feasible_solution, key=lambda x: objective_function(x[0], x[1], x[2], x[3]))
            best_fitness = objective_function(best_solution[0], best_solution[1], best_solution[2], best_solution[3])
        fitness_history.append(best_fitness)

        print(f"Generation{generation + 1}: Best Solution = {best_solution},Best Fitness = {best_fitness}")

    # Plot the fitness progress
    plt.plot(range(1, num_generation + 1), fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Genetic Algorithm - Fitness Progress")
    plt.show()

    return best_solution, best_fitness


population_size = 1000
num_generation = 100

best_solution, best_fitness = genetic_algorithm(population_size, num_generation)

if best_solution is not None:
    print("Final best solution:", best_solution)
    print("Final best fitness:", best_fitness)
else:
    print("No feasible solution found within the given constraints")