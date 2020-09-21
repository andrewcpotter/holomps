from star_circuit_JW_v2 import *
import numpy as np

bond_4_sweet_spot = [ 0.18571453,  6.2996646,   1.53252123,  3.92907394,  0.04571898,  5.52480991,
5.5524045,   1.98648086,  4.65813625,  5.73834787,  5.2805168, 2.35590066,
4.36276439,  5.66855415, 15.71212956,  5.58639466, -0.03465203,  3.92718468,
0.27985268,  6.10848261,  1.01198014,  5.48098847,  0.46659923, 2.86042039,
1.29931791,  5.49722482,  4.44790528,  2.51640997]

constant_list = [0.01,0.02,0.03,0.04,
0.02,0.01,0.05,0.06,
0.05,0.04,0.03,0.02,
0.03,0.05,0.02,0.03,
0.06,0.02,0.04,0.01,
0.04,0.05,0.02,0.03,
0.01,0.02]
params_per = 14
params_better_list = bond_4_sweet_spot[0: params_per] + constant_list[0: 13] + \
bond_4_sweet_spot[params_per: 2 * params_per] + constant_list[13: 26]
params_better = np.array(params_better_list)
(E1, filling_gs, sweet_spot) = fermi_hubbard_half_filling_star_circuit(1,8,0,True,100,20, True, params_better)
print(E1, filling_gs)
E0 = E1 - 1
iter = 1
while abs(E0-E1) > 1e-3:
    E0 = E1
    params_better = sweet_spot
    (E1, filling_gs, sweet_spot) = fermi_hubbard_half_filling_star_circuit(1,8,0,True,100,20, True, params_better)
    iter = iter + 1
    print(E1, filling_gs)
print(sweet_spot)
print(iter)