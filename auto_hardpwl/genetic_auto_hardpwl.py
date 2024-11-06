import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import json
import os
import argparse

ACT_FUNCS = {
    "swish": lambda x: x / (1.0 + np.exp(-x)) if -3.0 < x < 3.0 else (0 if x <= -3.0 else x),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)) if -3.0 < x < 3.0 else (0 if x <= -3.0 else 1.0),
    "tanh": lambda x: (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)) if -3.0 < x < 3.0 else (-1.0 if x <= -3.0 else 1.0),
    "gelu": lambda x: 0.5 * x * (1 + special.erf(x / np.sqrt(2))) if -3.0 < x < 3.0 else (0 if x <= -3.0 else x),
    "hswish": lambda x: x * np.clip(x + 3, 0, 6) / 6
}

def save_to_file(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)

def load_from_file(filename):
    with open(filename, "r") as file:
        return json.load(file)
    
def calculate_coeff_bias(a1, a2, act_type, bit):
    func = ACT_FUNCS[act_type]
    if a2 != a1:
        coeff = (func(a2) - func(a1)) / (a2 - a1)
    else:
    # Handle the case where a2 is equal to a1
        coeff = 0  # or some other appropriate value or handling method
    bias = -a1 * coeff + func(a1)
    resize = 2**(bit-2) # coeff is directly converted to fixed point format with the specified bit
    return (np.round(coeff*resize)/resize, np.round(bias*resize)/resize) 

def get_list(split_point, act_type, a_bit): 
    coeff_bias_pairs = [calculate_coeff_bias(a1, a2, act_type, a_bit) for a1, a2 in zip(split_point[:-1], split_point[1:])]
    coeff, bias = zip(*coeff_bias_pairs)
    return coeff, bias

def piecewise_linear_approximation(x, split_points, slopes, biases):
    index = np.digitize(x, split_points) - 1
    index = min(index, len(slopes) - 1)
    return slopes[index] * x + biases[index]

def compute_errors(func_original, func_approx, x_values):
    y_original = np.array([func_original(x) for x in x_values])
    y_approx = np.array([func_approx(x) for x in x_values])

    l1_loss = np.mean(np.abs(y_original - y_approx))
    l2_loss = np.sqrt(np.mean((y_original - y_approx)**2))
    mse = np.mean((y_original - y_approx)**2)

    return l1_loss, l2_loss, mse

def create_fixed_point_attr(decimal_bits, sp_range):
    scale_factor = 2**decimal_bits
    rand_val = np.random.uniform(sp_range[0], sp_range[1])
    return round(rand_val * scale_factor) / scale_factor

# Distributed Evolutionary Algorithms in Python
from deap import base, creator, tools, algorithms
# genetic algorithm for finding the best split points
def genetic_find_best_split_points(func_name, x_range, sp_range, num_splits, total_iters=1000, decimal_bits=5, pop_size=50, crossover_prob=0.7, mutation_prob=0.2):
    step = 0.01
    func = ACT_FUNCS[func_name]
    x_values = np.arange(x_range[0], x_range[1], step)
    y_values = np.array([func(x) for x in x_values])
    if "FitnessMin" not in creator.__dict__:
        # -1 represents to minimize the error
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        # Create the individual class
        creator.create("Individual", list, fitness=creator.FitnessMin)
    # Create the toolbox for different functions
    toolbox = base.Toolbox()
    # Generates values within the desired range with the specified decimal bits
    toolbox.register("attr_float", create_fixed_point_attr, decimal_bits, sp_range)
    # Initialize the individual class with the attr_float function
    # The initial values are all in fixed point format decided by the decimal_bits
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, num_splits) # Individual is a person
    # Initialize the population class with the individual class
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) # Population is a group of people
    # Define the mutation function
    def mutate_fixed_point(individual, mu, sigma, indpb, decimal_bits):
        scale_factor = 2**decimal_bits
        for i in range(len(individual)):
            if np.random.random() < indpb: # indpb: Independent probability for each attribute to be mutated
                # mu: mean, sigma: standard deviation
                individual[i] += np.random.normal(mu, sigma) # Add a random value from a normal distribution
                # Round the value to fixed point format with the specified decimal bits
                individual[i] = round(individual[i] * scale_factor) / scale_factor
                # Ensure that the individual stays within the specified range
                individual[i] = min(max(individual[i], sp_range[0]), sp_range[1])
        return individual,
    # Define the evaluation function
    def evaluate(individual):
        individual.sort()
        # -10000 and 10000 imitate the -inf and inf
        split_points = [-10000] + individual + [10000]
        coeff, bias = get_list(split_points, func_name, 8)
        approx_values = [piecewise_linear_approximation(x, split_points, coeff, bias) for x in x_values]
        # Calculate the mean squared error, could also use other error metrics
        error = np.mean((y_values - approx_values) ** 2)
        return error,
    # cxTwoPoint: crossover function for every two individuals whose probability > crossover_prob
    toolbox.register("mate", tools.cxTwoPoint) # cross probility is crossover_prob
    # mutation function
    toolbox.register("mutate", mutate_fixed_point, mu=0, sigma=0.2, indpb=0.1, decimal_bits=decimal_bits)
    # selection function
    toolbox.register("select", tools.selTournament, tournsize=3) # Randomly select 3 individuals and choose the best one
    # evaluation function
    toolbox.register("evaluate", evaluate)
    # construct the population with pop_size individuals
    population = toolbox.population(n=pop_size)
    # Imitate the evolution process
    algorithms.eaSimple(population, toolbox, crossover_prob, mutation_prob, total_iters)
    # Select the best individual
    best_individual = tools.selBest(population, 1)[0]
    best_splits = [-10000] + best_individual + [10000]
    best_splits.sort()
    return best_splits

def autopwl(activation_function_name, x_range=(-4, 4), num_splits=10, total_iters=100, decimal_bit=5, random=True):
    if activation_function_name not in ACT_FUNCS:
        print("Invalid activation function name. Valid names are:", ", ".join(ACT_FUNCS.keys()))
        return
    x_values = np.linspace(x_range[0], x_range[1], 1000)
    # force 2bit integer and 1bit sign
    sp_range = (-4+2**(-decimal_bit), 4-2**(-decimal_bit))
    split_points = genetic_find_best_split_points(activation_function_name, x_range, sp_range, num_splits, total_iters, decimal_bit)
    coeff, bias = get_list(split_points, activation_function_name, 8)
    y_approx = [piecewise_linear_approximation(val, split_points, coeff, bias) for val in x_values]
    # plot the approximation
    plt.figure(figsize=(8,6))
    plt.plot(x_values, [ACT_FUNCS[activation_function_name](x) for x in x_values], label=f"Original {activation_function_name}", color='blue')
    plt.plot(x_values, y_approx, '--', label=f"Approximation of {activation_function_name}", color='red')
    split_y = [ACT_FUNCS[activation_function_name](x) for x in split_points[1:-1]]
    plt.scatter(split_points[1:-1], split_y, color='green', label='Split Points')
    plt.legend()
    plt.title(f'{activation_function_name}_{num_splits}points_{decimal_bit}bits')
    plt.grid(True)
    # plt.show()
    save_dir = f'./pwl_plot/auto_pwl/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'./pwl_plot/auto_pwl/{activation_function_name}_{num_splits}points_{decimal_bit}bits.png', dpi=300)
    # plot the error
    return split_points, coeff, bias

def offline_pwlstore(act_func='hswish', decimal_bit_range=16, num_splits=7, total_iters=100, random=True):
    results = {}
    results[act_func] = {}
    x_values = np.linspace(-3.5, 3.5, 1000)  # Assuming -3.5 and 3.5 as default x_range
    func_original = ACT_FUNCS[act_func]
    for bit in range(1, decimal_bit_range+1):
        print(f"Start for activation_function: {act_func} and decimal_bit: {bit}")
        split_points, coeff, bias = autopwl(act_func, num_splits=num_splits, total_iters=total_iters, decimal_bit=bit, random=random)
        func_approx = lambda x: piecewise_linear_approximation(x, split_points, coeff, bias)
        l1_loss, l2_loss, mse = compute_errors(func_original, func_approx, x_values)
        results[act_func][bit] = {
            "split_points": split_points[1:-1],
            "coeff": coeff,
            "bias": bias,
            "L1_loss": l1_loss,
            "L2_loss": l2_loss,
            "MSE": mse
        }
        print(f"Done for activation_function: {act_func} and decimal_bit: {bit}")
        print(f"L1 Loss: {l1_loss}, L2 Loss: {l2_loss}, MSE: {mse}")
    save_to_file(results, f"./params_pwl/{act_func}_pwl.json")

def add_common_arguments(parser):
    parser.add_argument("--act_func", type=str, default='hswish', help="Activation function name")
    parser.add_argument("--num_splits", type=int, default=7, help="Number of split points")
    parser.add_argument("--total_iters", type=int, default=500, help="Total iterations")
    parser.add_argument("--decimal_bit", type=int, default=6, help="Decimal bit precision")
    parser.add_argument("--random", action="store_true", help="Use random initialization")
    return parser

def main():
    parser = argparse.ArgumentParser(description="Piecewise Linear Approximation Tool")
    subparsers = parser.add_subparsers(dest="command")
    # autopwl
    parser_autopwl = subparsers.add_parser('autopwl')
    parser_autopwl = add_common_arguments(parser_autopwl)
    # offline_pwlstore
    parser_offline = subparsers.add_parser('offline_pwlstore')
    parser_offline = add_common_arguments(parser_offline)
    parser_offline.add_argument("--decimal_bit_range", type=int, default=16, help="Decimal bit range")
    # compute_errors
    parser_compute = subparsers.add_parser('compute_errors')
    parser_compute = add_common_arguments(parser_compute)
    parser_compute.add_argument("--x_min", type=float, default=-3.5, help="Minimum value of x")
    parser_compute.add_argument("--x_max", type=float, default=3.5, help="Maximum value of x")
    parser_compute.add_argument("--split_points", nargs='+', type=float, help="List of split points")
    parser_compute.add_argument("--coeff", nargs='+', type=float, help="List of coefficients")
    parser_compute.add_argument("--bias", nargs='+', type=float, help="List of biases")

    args = parser.parse_args()

    if args.command == "autopwl":
        split_points, coeff, bias = autopwl(args.act_func, num_splits=args.num_splits, total_iters=args.total_iters, decimal_bit=args.decimal_bit, random=args.random)
        print(f"slope is :{coeff}, len is {len(coeff)}")
        print(f"intercept is :{bias}, len is {len(bias)}")
        print("seg_point is :", split_points[1:-1])
        print("len of seg_point is :", len(split_points[1:-1]))
        x_values = np.linspace(-3.5, 3.5, 1000)
        func_original = ACT_FUNCS[args.act_func]
        func_approx = lambda x: piecewise_linear_approximation(x, split_points, coeff, bias)
        l1_loss, l2_loss, mse = compute_errors(func_original, func_approx, x_values)
        print(f"L1 Loss: {l1_loss}")
        print(f"L2 Loss: {l2_loss}")
        print(f"MSE: {mse}")   

    elif args.command == "offline_pwlstore":
        offline_pwlstore(act_func=args.act_func, decimal_bit_range=args.decimal_bit_range, num_splits=args.num_splits, total_iters=args.total_iters, random=args.random)
    # Some bug exists in compute_errors
    elif args.command == "compute_errors":
        x_values = np.linspace(-3.5, 3.5, 1000)
        func_original = ACT_FUNCS[args.act_func]
        split_points = [-10000] + args.split_points + [10000]
        func_approx = lambda x: piecewise_linear_approximation(x, split_points, args.coeff, args.bias)

        l1_loss, l2_loss, mse = compute_errors(func_original, func_approx, x_values)
        print(f"L1 Loss: {l1_loss}")
        print(f"L2 Loss: {l2_loss}")
        print(f"MSE: {mse}")   

if __name__ == "__main__":
    np.random.seed(42)
    main()

