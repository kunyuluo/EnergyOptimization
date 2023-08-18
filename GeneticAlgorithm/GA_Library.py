import numpy as np
from geneticalgorithm import geneticalgorithm as ga


# Define the equations
def fitness_function(variables):
    x1 = variables[0]
    x2 = variables[1]
    x3 = variables[2]

    # Apply Constraints
    penalty = 0
    if 5*x1 + 7*x2 + 4*x3 > 10:
        penalty = np.inf

    return -(16 * x1 + 22 * x2 + 12 * x3) + penalty


algorithm_params = {'max_num_iteration': None, 'population_size': 100, 'mutation_probability': 0.5, 'elit_ratio': 0.01,
                    'crossover_probability': 0.5, 'parents_portion': 0.3, 'crossover_type': 'uniform',
                    'max_iteration_without_improv': None}

# Create an instance of the GA solver
model = ga(function=fitness_function, dimension=3, variable_type='bool', algorithm_parameters=algorithm_params)
model.run()
