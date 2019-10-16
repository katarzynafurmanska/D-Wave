from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import pandas as pd

sapi_token = 'DEV-51a6008265b365518036cc29683e84870c667e63'
dwave_url = 'https://cloud.dwavesys.com/sapi'

DWaveSamplerInstance = DWaveSampler(token = sapi_token, endpoint = dwave_url, solver = 'DW_2000Q_2_1')

###############################################################################################################################################################################################
# simple binary function
from pyqubo import Binary

s1, s2, s3, s4 = Binary("s1"), Binary("s2"), Binary("s3"), Binary("s4")
H = (4 * s1 + 2 * s2 + 7 * s3 + s4) ** 2

model = H.compile()
Q, offset = model.to_qubo()
print(Q)

response = EmbeddingComposite(DWaveSamplerInstance).sample_qubo(Q, num_reads=1000)
for sample in response.data():
    print(sample[0], "Energy: ", sample[1], "Occurrences: ",sample[2])