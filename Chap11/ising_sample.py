from sampling import SASampling
import numpy as np
import random

h = {'a':-1}
J = {('a','b'):1, ('b','c'):1, ('c','d'):1}



sampler = SASampling()
sampler.sample_ising(h, J, n_iter=100)

print("optimial result", sampler.sample)
