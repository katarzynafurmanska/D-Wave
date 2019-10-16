from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from _tokens import get_sapi_token, get_dwave_url

sapi_token = get_sapi_token()
dwave_url = get_dwave_url()

# create instance of DWaveSampler
DWaveSamplerInstance = DWaveSampler(token = sapi_token, endpoint = dwave_url, solver = 'DW_2000Q_2_1')
###############################################################################################################################################################################################
### NOT gate
### z:= ~x
### H(x,z) = 2*x*z - x - z - 1
linear = {(0, 0): -1, (1, 1): -1}
quadratic = {(0, 1): 2}

Q = dict(linear)
Q.update(quadratic)

response = EmbeddingComposite(DWaveSamplerInstance).sample_qubo(Q, num_reads=1000)
for sample in response.data():
    print(sample[0], "Energy: ", sample[1], "Occurrences: ",sample[2])


###############################################################################################################################################################################################
### AND gate -> NOT gate
### x2:= x0 and x1, x3 := ~x2
### H1(x0, x1, x2) := x0*x1 + 2(x0 + x1)*x2 + 3*x2
### H2(x2, x3) := 2*x2*x3 - x2 - x3 -1
### H := H1 + H2 = 2*x2 - x3 + x0*x1 + 2*x0*x2 + 2*x1*x2 + 2*x2*x3

### ? it doesn't return one solution - x3=0
linear = {(2, 2): 2, (3, 3): -1}
quadratic = {(0, 1): 1, (0, 2): 2, (1, 2): 2, (2, 3): 2}

Q = dict(linear)
Q.update(quadratic)

response = EmbeddingComposite(DWaveSamplerInstance).sample_qubo(Q, num_reads=1000)
for sample in response.data():
    print(sample[0], "Energy: ", sample[1], "Occurrences: ",sample[2])
