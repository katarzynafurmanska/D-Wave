from pyqubo import Array, Placeholder, solve_qubo, Constraint, Sum
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import os
import warnings
from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category = MatplotlibDeprecationWarning)
import matplotlib
matplotlib.use('Agg')
from help_func.plot_city import plot_city
from help_func.dist import dist
path = os.getcwd()

###############################################################################################################################################################################################
# TSP problem
# see https://github.com/recruit-communications/pyqubo/blob/master/notebooks/TSP.ipynb
###############################################################################################################################################################################################

# City names and coordinates list[(index, (x, y))]
cities = {
    0: (0, 0),
    1: (1, 3),
    2: (3, 2),
    3: (2, 1),
    4: (0, 1)
}

# Plot cities
plot_city(cities)
plt.savefig(path + '/venv/results/test/test_simulator_empty.png')
plt.close()

n_city = len(cities)
# x is array of binary variables c[i][j] = 1 when city j is visited at time i and 0 otherwise
x = Array.create('c', (n_city, n_city), 'BINARY')

# Constraint not to visit more than two cities at the same time.
time_const = 0.0
for i in range(n_city):
    time_const += Constraint((Sum(0, n_city, lambda j: x[i, j]) - 1)**2, label="time{}".format(i))

# Constraint not to visit the same city more than twice.
city_const = 0.0
for j in range(n_city):
    city_const += Constraint((Sum(0, n_city, lambda i: x[i, j]) - 1)**2, label="city{}".format(j))

# distance of route
distance = 0.0
for i in range(n_city):
    for j in range(n_city):
        for k in range(n_city):
            d_ij = dist(i, j, cities)
            distance += d_ij * x[k, i] * x[(k+1)%n_city, j]


# Construct hamiltonian
A = Placeholder("A")
H = distance + A * (time_const + city_const)

# Compile model
model = H.compile()

# Generate QUBO
feed_dict = {'A': 4.0}
qubo, offset = model.to_qubo(feed_dict=feed_dict)

# solve QUBO with Simulated Annealing (SA) provided by neal.
sol = solve_qubo(qubo)

# decode solution, constraints that are broken and energy value
solution_sim, broken, energy = model.decode_solution(sol, vartype="BINARY", feed_dict=feed_dict)

plot_city(cities, solution_sim["c"])
plt.savefig(path + '/venv/results/test/test_simulator.png')
plt.close()

###############################################################################################################################################################################################
# solve TSP problem on D-Wave machine
###############################################################################################################################################################################################

from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

sapi_token = 'DEV-51a6008265b365518036cc29683e84870c667e63'
dwave_url = 'https://cloud.dwavesys.com/sapi'

DWaveSamplerInstance = DWaveSampler(token = sapi_token, endpoint = dwave_url, solver = 'DW_2000Q_2_1')

###############################################################################################################################################################################################
# sample and solve problem
response = EmbeddingComposite(DWaveSamplerInstance).sample_qubo(qubo, num_reads=1000)

# extract results
results = pd.DataFrame.from_records(list(response.data()), columns = ["vars", "energy", "occurrences","chain_break_fraction"]).sort_values("occurrences", ascending=False)

# decode solution
results["visited (time:city)"] = results["vars"].apply(lambda x:  str([key[2] + ':' + key[5] for (key, val) in x.items() if val == 1]))
results["no of visited cities"] = results["vars"].apply(lambda x:  sum([val for (key, val) in x.items()]))

# select solution with exactly the same number of visited cities as their amount
# there still can be some duplicates
results = results[results["no of visited cities"]==len(cities.keys())]
results = results[["visited (time:city)", "occurrences", "energy", "vars"]].sort_values("energy", ascending=True)

results_consolidated = results.pivot_table(values = 'occurrences', index = ["visited (time:city)"], aggfunc = np.sum)
solution_coded = results["vars"][results["visited (time:city)"] == results_consolidated.index[0]].iloc[0]

solution_dwave, broken, energy = model.decode_solution(solution_coded, vartype="BINARY", feed_dict=feed_dict)

# Plot solution
plot_city(cities, solution_dwave["c"])
plt.savefig(path + '/venv/results/test/test_dwave.png')
plt.close()