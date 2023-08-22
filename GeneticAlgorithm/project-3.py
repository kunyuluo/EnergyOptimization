import random
import matplotlib.pyplot as plt


# Define the objective function
def objective_function(x1):
    return pow(x1, 2) - 2*x1 + 2


# Define constraints
def constraint1(x1):
    return -2 <= x1 <= 10


def genetic_algorithm(population_size, num_generation):

    best_solution = None
    best_fitness = None
    fitness_history = []

    # Initialize
    population = []
    for i in range(population_size):
        x1 = random.uniform(-2, 5)
        population.append(x1)

    for generation in range(num_generation):
        # Evaluate
        fitness = [objective_function(x1) for x1 in population]
        valid_population = [x for x in population if constraint1(x)]

        # Selection
        if valid_population:
            valid_fitness = [objective_function(x) for x in valid_population]
            parents = random.choices(valid_population, weights=valid_fitness, k=population_size)
        else:
            parents = []
            while len(parents) < population_size:
                potential_parent = random.choice(population)
                if constraint1(potential_parent):
                    parents.append(potential_parent)

        # Crossover
        offsprings = []
        for i in range(population_size):
            parent1, parent2 = random.choices(parents, k=2)
            x_child = random.choice([parent1, parent2])
            offsprings.append(x_child)

        # Mutation
        mutation_rate = 1/(generation+1)
        for i in range(population_size):
            if random.random() < mutation_rate:
                offsprings[i] = random.uniform(-2, 5)

        # Elitism
        if best_solution is not None:
            offsprings[0] = best_solution

        population = offsprings

        # Find the best solution
        valid_solutions = [x for x in population if constraint1(x)]
        if valid_solutions:
            best_solution = min(valid_solutions, key=lambda x: objective_function(x))
            best_fitness = objective_function(best_solution)
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