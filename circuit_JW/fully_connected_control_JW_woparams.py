from full_connected_JW_v2 import *
import numpy as np
(E1, filling_gs, sweet_spot) = fermi_hubbard_half_filling_fully_connected(1,8,0,True,100,20, False, 1)
print(E1, filling_gs)
E0 = E1 - 1
iter = 1
while abs(E0-E1) > 1e-3:
    E0 = E1
    params_better = sweet_spot
    (E1, filling_gs, sweet_spot) = fermi_hubbard_half_filling_fully_connected(1,8,0,True,100,20, True, params_better)
    iter = iter + 1
    print(E1, filling_gs)
print(sweet_spot)
print(iter)