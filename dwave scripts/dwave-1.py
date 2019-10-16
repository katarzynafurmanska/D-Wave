from dwave.system.samplers import DWaveSampler
from _tokens import get_sapi_token, get_dwave_url

sapi_token = get_sapi_token()
dwave_url = get_dwave_url()

# objective function: H(a,b,c) = -a-b-c-2ab-2bc-2ac; see https://docs.dwavesys.com/docs/latest/c_gs_7.html#em
# we look for values of a,b,c such that H(a,b,c) is minimized

### Manual embedding
# Set Q for the minor-embedded problem QUBO
qubit_biases = {(0, 0): 0.3333, (1, 1): -0.333, (4, 4): -0.333, (5, 5): 0.333}
coupler_strengths = {(0, 4): 0.667, (0, 5): -1, (1, 4): 0.667, (1, 5): 0.667}
Q = dict(qubit_biases)
Q.update(coupler_strengths)

# create instance of DWaveSampler
DWaveSamplerInstance = DWaveSampler(token = sapi_token, endpoint = dwave_url, solver='DW_2000Q_2_1')
# check active nodes (qubits) and edges (couplers)
print(DWaveSamplerInstance.nodelist)
print(DWaveSamplerInstance.edgelist)

# Sample once on a D-Wave system and print the returned sample
response = DWaveSamplerInstance.sample_qubo(Q, num_reads=1000)
for sample in response.data():
    print(sample[0], "Energy: ", sample[1], "Occurrences: ",sample[2])
















###############################################################################################################################################################################################
### Automatic embedding
from dwave.system.composites import EmbeddingComposite
# Set Q for the problem QUBO
linear = {('q1', 'q1'): -1, ('q2', 'q2'): -1, ('q3', 'q3'): -1}
quadratic = {('q1', 'q2'): 2, ('q1', 'q3'): 2, ('q2', 'q3'): 2}
Q = dict(linear)
Q.update(quadratic)
# Minor-embed and sample 1000 times on a default D-Wave system
response = EmbeddingComposite(DWaveSamplerInstance).sample_qubo(Q, num_reads=1000)
for sample in response.data():
    print(sample[0], "Energy: ", sample[1], "Occurrences: ",sample[2])
















