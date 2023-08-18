import random
import matplotlib.pyplot as plt


# Define the objective function
def objective_function(x1, x2, x3):
    return 16 * x1 + 22 * x2 + 25 * x3


# Define constraints
def constraint1(x1, x2, x3):
    return 5 * x1 + 7 * x2 + 4 * x3 <= 15


def genetic_algorithm(population_size, num_generation):

    best_solution = None
    best_fitness = float("-inf")
    fitness_history = []

    # Initialization
    population = []

    for i in range(population_size):
        x1 = random.randint(0, 1)
        x2 = random.randint(0, 1)
        x3 = random.randint(0, 1)
        population.append((x1, x2, x3))

    for generation in range(num_generation):
        # Evaluation
        fitness = [objective_function(x1, x2, x3) for x1, x2, x3 in population]
        valid_population = [(x1, x2, x3) for x1, x2, x3 in population if constraint1(x1, x2, x3)]

        # Selection
        if valid_population:
            valid_fitness = [objective_function(x1, x2, x3) for x1, x2, x3 in valid_population]
            parents = random.choices(valid_population, weights=valid_fitness, k=population_size)
        else:
            parents = []
            while len(parents) < population_size:
                potential_parent = random.choices(population, weights=fitness)
                if constraint1(potential_parent[0], potential_parent[1], potential_parent[2]):
                    parents.append(potential_parent)
        # Crossover
        offspring = []
        for i in range(population_size):
            parent1, parent2, parent3 = random.choices(parents, k=3)
            x1_child = random.choice([parent1[0], parent2[0], parent3[0]])
            x2_child = random.choice([parent1[1], parent2[1], parent3[1]])
            x3_child = random.choice([parent1[2], parent2[2], parent3[2]])
            offspring.append((x1_child, x2_child, x3_child))

        # Mutation
        mutation_rate = 1/(generation+1)
        for i in range(population_size):
            if random.random() < mutation_rate:
                offspring[i] = (random.randint(0, 1), random.randint(0, 1), random.randint(0, 1))

        # Elitism
        if best_solution is not None:
            offspring[0] = best_solution

        population = offspring

        # Find the best solution
        valid_solution = [(x1, x2, x3) for x1, x2, x3 in population if constraint1(x1, x2, x3)]
        if valid_solution:
            best_solution = max(valid_solution, key=lambda x: objective_function(x[0], x[1], x[2]))
            best_fitness = objective_function(best_solution[0], best_solution[1], best_solution[2])
        fitness_history.append(best_fitness)

        print(f"Generation{generation+1}: Best Solution = {best_solution},Best Fitness = {best_fitness}")

    # Plot the fitness progress
    plt.plot(range(1, num_generation + 1), fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Genetic Algorithm - Fitness Progress")
    plt.show()

    return best_solution, best_fitness


population_size = 1000
num_generation = 50

best_solution, best_fitness = genetic_algorithm(population_size, num_generation)

if best_solution is not None:
    print("Final best solution:", best_solution)
    print("Final best fitness:", best_fitness)
else:
    print("No feasible solution found within the given constraints")

# x1 = random.randint(0, 1)
# print(x1)
