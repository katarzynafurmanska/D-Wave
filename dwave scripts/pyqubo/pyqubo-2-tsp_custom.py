from pyqubo import Array, Placeholder, solve_qubo, Constraint, Sum
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import csv
import time
import pickle
import os
import warnings
from matplotlib.cbook.deprecation import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category = MatplotlibDeprecationWarning)
from help_func.plot_city import plot_city
from help_func.dist import dist
import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
path = os.getcwd()

###############################################################################################################################################################################################
# TSP problem for countries cities
# data from: http://www.math.uwaterloo.ca/tsp/world/countries.html
###############################################################################################################################################################################################

# select country
#country = 'western_sahara'
country = 'luxembourg'
#country = 'test_temp'

city_data = path + '/venv/data/' + country + '.csv'
with open(city_data, mode='r') as infile:
    reader = csv.reader(infile, delimiter='\t')
    cities = {index:(float(row[1]),float(row[2])) for index, row in enumerate(reader)}

#plt = plot_city(cities)
#plt.savefig(path + '/venv/results/' + country + '/' + country + '_empty.png')
#plt.close()

n_city = len(cities)
x = Array.create('c', (n_city, n_city), 'BINARY')

############### this part is time-consuming - result calculated can be downloaded below
###### distance expression for 29 cities take approx 8 mins to calculate
# Constraint not to visit more than two cities at the same time.
start_time1 = time.time()
time_const1 = 0.0
for i in range(n_city):
    time_const1 += Constraint((Sum(0, n_city, lambda j: x[i, j]) - 1)**2, label="time{}".format(i))
end_time1 = time.time()
print("1: " + str(end_time1 - start_time1))

start_time2 = time.time()
time_const2 = sum([Constraint((Sum(0, n_city, lambda j: x[i, j]) - 1)**2, label="time{}".format(i)) for i in range(n_city)])
end_time2 = time.time()
print("2: " + str(end_time2 - start_time2))

def process(i,j,k):
    #if j == 0 & k == 0:
    #    print("i: " + str(i) + ", j: " + str(j) + ", k " + str(k))
    return dist(i, j, cities) * x[k, i] * x[(k+1)%n_city, j]

from functools import reduce
start_time3 = time.time()
distance = [process(i,j,k) for i in range(n_city) for j in range(n_city) for k in range(n_city)]
time_const3 = reduce(lambda a,b : a+b, distance)
end_time3 = time.time()
print("3: " + str(end_time3 - start_time3))

# Constraint not to visit the same city more than twice.
city_const = sum([Constraint((Sum(0, n_city, lambda i: x[i, j]) - 1)**2, label="city{}".format(j)) for i in range(n_city)])


distance = [process(i,j,k) for i in range(n_city) for j in range(n_city) for k in range(n_city)]







# distance of route
distance = sum([dist(i, j, cities) * x[k, i] * x[(k+1)%n_city, j] for i in range(n_city) for j in range(n_city) for k in range(n_city)])


n_city = 3
# distance of route
distance = 0.0
start_time = time.time()
for i in range(n_city):
    for j in range(n_city):
        for k in range(n_city):
            d_ij = dist(i, j, cities)
            distance += d_ij * x[k, i] * x[(k+1)%n_city, j]
            print("distance: " + str((i-1)*j*k+(j-1)*k+k) + '/' + str(n_city*n_city*n_city))
end_time = time.time()





print("Time: " + str(round((end_time - start_time)/60,2)) + "[min]")

# Construct hamiltonian
A = Placeholder("A")
H = distance + A * (time_const + city_const)

# Compile model
model = H.compile()
#pickle.dump(model, open(path +  '/venv/pickles/' + country + '.p', 'wb' ) )

############### load model calculated earlier
model = pickle.load(open(path +  '/venv/pickles/' + country + '.p', "rb" ) )

# Generate QUBO
A_val = 5       # change me!
feed_dict = {'A': A_val}
qubo, offset = model.to_qubo(feed_dict=feed_dict)

# solve QUBO with Simulated Annealing (SA) provided by neal.
sol = solve_qubo(qubo)

solution_sim, broken, energy = model.decode_solution(sol, vartype="BINARY", feed_dict=feed_dict)

plt = plot_city(cities, solution_sim["c"])
plt.savefig(path + '/venv/results/' + country + '/' + country + '_simulator_' + str(A_val) + '.png')
plt.close()

###############################################################################################################################################################################################
# solve TSP problem on D-Wave machine
###############################################################################################################################################################################################
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite

# select solver
#solver = 'DW_2000Q_2_1'
solver = 'DW_2000Q_5'

sapi_token = 'DEV-51a6008265b365518036cc29683e84870c667e63'
dwave_url = 'https://cloud.dwavesys.com/sapi'

DWaveSamplerInstance = DWaveSampler(token = sapi_token, endpoint = dwave_url, solver = solver)

###############################################################################################################################################################################################
start_time = time.time()
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
end_time = time.time()
print("Time: " + str(round((end_time - start_time)/60,2)) + "[min]")

# Plot solution
plt = plot_city(cities, solution_dwave["c"])
plt.savefig(path + '/venv/results/' + country + '/' + country + '_dwave_' + solver + '_' + str(A_val) + '.png')
plt.close()
