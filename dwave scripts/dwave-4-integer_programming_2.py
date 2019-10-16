from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from sympy import symbols, Poly
import pandas as pd
import numpy as np
import datetime
import itertools
import random
from _tokens import get_sapi_token, get_dwave_url

sapi_token = get_sapi_token()
dwave_url = get_dwave_url()

DWaveSamplerInstance = DWaveSampler(token = sapi_token, endpoint = dwave_url, solver = 'DW_2000Q_2_1')

###############################################################################################################################################################################################
# maximum available case found: 2000 qubits, 9s
k = 2000
Q = dict(zip([(i,i) for i in range(1,k+1)], np.ones(k)))

print(datetime.datetime.now())
response = EmbeddingComposite(DWaveSamplerInstance).sample_qubo(Q, num_reads=1000)
for sample in response.data():
    print(sample[0], "Energy: ", sample[1], "Occurrences: ",sample[2])
print(datetime.datetime.now())

###############################################################################################################################################################################################
# maximum available case found: 2000 qubits, random values - failed during initialization. embeddings may be invalid.
k = 2000
Q = {t: random.uniform(-1, 1) for t in itertools.product(range(k), repeat=2)}

print(datetime.datetime.now())
response = EmbeddingComposite(DWaveSamplerInstance).sample_qubo(Q, num_reads=1000)
for sample in response.data():
    print(sample[0], "Energy: ", sample[1], "Occurrences: ",sample[2])
print(datetime.datetime.now())

# error - no embedding found
k = 2050
Q = dict(zip([(i,i) for i in range(1,k+1)], np.ones(k)))

print(datetime.datetime.now())
response = EmbeddingComposite(DWaveSamplerInstance).sample_qubo(Q, num_reads=1000)
for sample in response.data():
    print(sample[0], "Energy: ", sample[1], "Occurrences: ",sample[2])
print(datetime.datetime.now())
