from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
from sympy import symbols, Poly
import pandas as pd
import numpy as np
from _tokens import get_sapi_token, get_dwave_url

sapi_token = get_sapi_token()
dwave_url = get_dwave_url()

DWaveSamplerInstance = DWaveSampler(token = sapi_token, endpoint = dwave_url, solver = 'DW_2000Q_2_1')

###############################################################################################################################################################################################
def reduce_squares(polynomial, symb):
    Coeffs = [coeff for (nums, coeff) in polynomial.terms() if any(nums)!=0]

    n_list = []

    for i in range(len(polynomial.terms()[0][0])):
        n = tuple([1 if nums[i] == 2 else nums[i] for (nums, coeff) in polynomial.terms()])
        n_list.append(n)

    new_terms = tuple(zip(tuple(zip(*n_list)), Coeffs))
    Poly(sum([np.prod([x**n for (x,n) in zip(symb, nums)])*coeff for (nums, coeff) in new_terms]))
    new_polynomial = Poly(sum([np.prod([x**n for (x,n) in zip(symb, nums)])*coeff for (nums, coeff) in new_terms]))

    return new_polynomial

def translate_indexes(polynomial):
    translate_dict = {}

    # create tuples of type (n,n) for linear terms and (n,m) for quadratic terms
    for polynomial_index in polynomial.as_dict().keys():
        d = dict(enumerate(polynomial_index, 1))
        qubit_index = tuple([key for key, value in d.items() if value == 1])

        # for linear terms, tuple of type (n,) needs to be transformed to (n,n)
        if len(qubit_index) == 1:
            #qubit_index = qubit_index + qubit_index
            translate_dict[polynomial_index] = tuple(qubit_index + qubit_index)
        else:
            translate_dict[polynomial_index] = qubit_index

    qubit_dict = {}

    for polynomial_index in polynomial.as_dict().keys():
        qubit_index = translate_dict[polynomial_index]
        qubit_dict[qubit_index] = int(polynomial.as_dict()[polynomial_index])
    return qubit_dict

###############################################################################################################################################################################################
### Problem: 2x1 + 3x2 -> min
###          x1 + x2   = 1
###          x1,x2 - bin
### H = 2x1 + 3x2 + P*(x1 + x2 - 1)^2
### for P = 5: H = -3x1 - 2x2 + 10x1x2

x1,x2 = symbols('x1:3')
obj = Poly(2*x1 + 3*x2)         # ----> min
c1 = Poly(x1 + x2 - 1)          # constraint
P = 5      # 0.5                # penalty
H = obj + P*c1**2

# reduce x**2 to x as x is binary
H1 = reduce_squares(H, [x1,x2])
Q = translate_indexes(H1)

response = EmbeddingComposite(DWaveSamplerInstance).sample_qubo(Q, num_reads=1000)
for sample in response.data():
    print(sample[0], "Energy: ", sample[1], "Occurrences: ",sample[2])

###############################################################################################################################################################################################
### Problem: 2x1 - 3x2 -> min
###          x1 + x2  <= 1 ----> x1 + x2 + s = 1, s - bin
###          x1,x2 - bin
### H = 2x1 - 3x2 + P*(x1 + x2 + s - 1)^2
### for P = 1: x1 - 4x2 - s + 2s*x1 + 2s*x2 + 2x1x2 + 1

x1,x2,s = symbols('x1:3,s')

obj = Poly(2*x1 - 3*x2)             # ----> min
c1 = Poly(x1 + x2 + s - 1)          # constraint
P = 1                               # penalty
H = obj + P*c1**2

# reduce x**2 to x as x is binary
H1 = reduce_squares(H, (x1,x2,s))
Q = translate_indexes(H1)

response = EmbeddingComposite(DWaveSamplerInstance).sample_qubo(Q, num_reads=1000)
for sample in response.data():
    print(sample[0], "Energy: ", sample[1], "Occurrences: ",sample[2])


###############################################################################################################################################################################################
### Problem: 2x1 - 3x2 -> min
###          2x1 + x2 <= 3 -----> 2x1 + x2 + s = 3, s<=3
###          x1 + x2   = 1
###          x1, x2 - bin; s - int -> s = 2s1 + s2, s1,s2 - bin
### H = 2x1 - 3x2 + P1*(2x1 + x2 + 2s1 + s2 - 3)^2 + P2*(x1 + x2 - 1)^2

x1,x2,s1,s2 = symbols('x1:3,s1:3')

obj = Poly(2*x1 - 3*x2)                     # ----> min
c1 = Poly(2*x1 + x2 + 2*s1 + s2 - 3)        # constraint 1
c2 = Poly(x1 + x2 - 1)                      # constraint 2
P1 = 1                                      # penalty 1
P2 = 1                                      # penalty 2
H = obj + P1*c1**2 + P2*c2**2

# reduce x**2 to x as x is binary
H1 = reduce_squares(H, (x1,x2,s1,s2))
Q = translate_indexes(H1)

response = EmbeddingComposite(DWaveSamplerInstance).sample_qubo(Q, num_reads=1000)
for sample in response.data():
    print(sample[0], "Energy: ", sample[1], "Occurrences: ",sample[2])

# with P=2 it gives wrong result


################################################################################################################################
#                                         S  O  L  V  E  R     T  Y  P  E  S
################################################################################################################################

from dwave.cloud import Client
client = Client.from_config(token=sapi_token)
client.get_solvers()        #[Solver(id='DW_2000Q_5'), Solver(id='DW_2000Q_2_1')]
from dwave.system.samplers import DWaveSampler
sampler =  DWaveSampler(token = sapi_token, endpoint = dwave_url, solver = 'DW_2000Q_2_1')
sampler.parameters

################################################################################################################################
#                                             S  I  M  U  L  A  T  O  R
################################################################################################################################

import dimod

# cannot put no of repetitions
#solver = dimod.ExactSolver().sample(bqm)
#solver = dimod.SimulatedAnnealingSampler()
#response = solver.sample_qubo(Q)

bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
#response = dimod.ExactSolver().sample(bqm)     # other solver simulator?
response = dimod.SimulatedAnnealingSampler().sample(bqm, num_reads = 10)
#for sample, energy, num_occurrences in response.data(['sample', 'energy', 'num_occurrences']):
#    print("sample: ", sample, "energy: ", energy, "num_occurences: ", num_occurrences)

# rearrange the response
response_df = pd.DataFrame([(str(sample), energy, num_occurrences) for sample, energy, num_occurrences in response.data(['sample', 'energy', 'num_occurrences'])])
response_df.columns = ['sample', 'energy', 'num_occurrences']
response_pv = response_df.pivot_table(index = ['sample','energy'], values = 'num_occurrences', aggfunc = sum)
print(response_pv.sort_values(by = 'num_occurrences', ascending = False))

###############################################################################################################################################################################################
a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12 = symbols('a0:13')
b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12 = symbols('b0:13')

obj = Poly(2*(a0+2*a1+4*a2+8*a3+16*a4+32*a5+64*a6+128*a7+256*a8+512*a9+1024*a10+2048*a11+4096*a12) - b0+2*b1+4*b2+8*b3+16*b4+32*b5+64*b6+128*b7+256*b8+512*b9+1024*b10+2048*b11+4096*b12)   # ----> min
c1 = Poly(a0+2*a1+4*a2+8*a3+16*a4+32*a5+64*a6+128*a7+256*a8+512*a9+1024*a10+2048*a11+4096*a12 - 4000)        # constraint 1
c2 = Poly(b0+2*b1+4*b2+8*b3+16*b4+32*b5+64*b6+128*b7+256*b8+512*b9+1024*b10+2048*b11+4096*b12 - 4000)        # constraint 2
P1 = 10                                      # penalty 1
P2 = 10                                      # penalty 2
H = obj + P1*c1**2 + P2*c2**2

# reduce x**2 to x as x is binary
H1 = reduce_squares(H, (a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12))
Q = translate_indexes(H1)

response = EmbeddingComposite(DWaveSamplerInstance).sample_qubo(Q, num_reads=1000)
for sample in response.data():
    print(sample[0], "Energy: ", sample[1], "Occurrences: ",sample[2])
