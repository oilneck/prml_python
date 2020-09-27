from sampling import SASampling
import numpy as np

h = {'a':-10}# external field
J = {('a','b'):1, ('b','c'):1, ('c','d'):1}# exchange interaction

sampler = SASampling()
result = sampler.sample_ising(h, J, n_iter=100)

print("optimum result", result)
