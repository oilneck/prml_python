import numpy as np

class Solver(object):

    def __init__(self):
        pass

    def bisect(self, func_f, x_min, x_max, error:float=1e-5 , max_iter:int=100):
        num_calc = 0

        while(True):
            x_mid = (x_max + x_min) / 2.

            if (0.0 < func_f(x_mid) * func_f(x_max)):
                x_max = x_mid
            else:
                x_min = x_mid

            num_calc += 1

            if(np.allclose(x_max, x_min, atol=error) or max_iter <= num_calc):
                break

        return x_mid
