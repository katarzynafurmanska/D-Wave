from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import re
import pandas as pd
from _tokens import get_sapi_token, get_dwave_url

sapi_token = get_sapi_token()
dwave_url = get_dwave_url()

# create instance of DWaveSampler
DWaveSamplerInstance = DWaveSampler(token = sapi_token, endpoint = dwave_url, solver='DW_2000Q_2_1')
###############################################################################################################################################################################################
### Single region, 4 colours
### only 1 colour can be selected: q1 + q2 + q3 + q4 =1
### H = -q1 - q2 - q3 - q4 + 2*(q1*q2 + q2*q3 + q3*q4 + q1*q3 + q1*q4 + q2*q4)

linear = {key: -1 for key in [('q' + str(color), 'q' + str(color)) for color in range(1,5)]}
quadratic = {key: 2 for key in [('q' + str(color1), 'q' + str(color2)) for color1 in range(1,5) for color2 in range(2,5) if color1<color2]}
Q = dict(linear)
Q.update(quadratic)

response = EmbeddingComposite(DWaveSamplerInstance).sample_qubo(Q, num_reads=1000)
for sample in response.data():
    print(sample[0], "Energy: ", sample[1], "Occurrences: ",sample[2])











###############################################################################################################################################################################################
### Two regions: A,B; 2 colours: 1,2
### 1: only 1 colour per region can be selected, 2: two regions cannot have the same colour
### H1 = -qA1 - qA2 + 2*qA1*qA2 - qB1 - qB2 + 2*qB1*qB2       #qA1 = region A has colour 1
### H2 = -qA1 - qB1 + 2*qA1*qB1 - qA2 - qB2 + 2*qA2*qB2
### H = H1 + H2 = -2*qA1 -2*qA2 -2*qB1 - 2*qB2 + 2*qA1*qA2 + 2*qB1*qB2 + 2*qA1*qB1 + 2*qA2*qB2

c_linear = {key: -2 for key in [('q' + region + str(color), 'q' + region + str(color)) for region in ['A', 'B'] for color in range(1,3)]}
c_quadratic1 = {key: 2 for key in [('q' + region + str(color1), 'q' + region + str(color2)) for region in ['A', 'B'] for color1 in range(1,5) for color2 in range(2,3) if color1<color2]}
c_quadratic2 = {key: 2 for key in [('q' + region1 + str(color), 'q' + region2 + str(color)) for region1 in ['A', 'B'] for region2 in ['A', 'B'] for color in range(1,3) if region1<region2]}

Q = dict(c_linear)
Q.update(c_quadratic1)
Q.update(c_quadratic2)

response = EmbeddingComposite(DWaveSamplerInstance).sample_qubo(Q, num_reads=1000)
for sample in response.data():
    print(sample[0], "Energy: ", sample[1], "Occurrences: ",sample[2])









