import numpy as np
from geneticalgorithm import geneticalgorithm as ga


# Define the equations
def fitness_function(variables):
    x1 = variables[0]
    x2 = variables[1]
    x3 = variables[2]

    # Apply Constraints
    penalty = 0
    if 5 * x1 + 7 * x2 + 4 * x3 > 10:
        penalty = np.inf

    return -(16 * x1 + 22 * x2 + 12 * x3) + penalty


def fitness_function_2(variables):
    x1 = variables[0]

    # Apply Constraints
    penalty = 0
    if x1 < -2 or x1 > 5:
        penalty = np.inf

    return pow(x1, 2) - 2 * x1 + 2 + penalty


varbound = np.array([[-2, 5]])

algorithm_params = {'max_num_iteration': None, 'population_size': 1000, 'mutation_probability': 0.5, 'elit_ratio': 0.01,
                    'crossover_probability': 0.5, 'parents_portion': 0.3, 'crossover_type': 'uniform',
                    'max_iteration_without_improv': None}

# Create an instance of the GA solver
model = ga(function=fitness_function_2, dimension=1, variable_type='real', algorithm_parameters=algorithm_params,
           variable_boundaries=varbound)
model.run()
