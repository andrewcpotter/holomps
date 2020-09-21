from More_entanglement_v2 import fermi_hubbard_half_filling_more_entanglement
import numpy as np
params_better_list = [0.18571453,  6.2996646,   1.53252123,  3.92907394,  0.04571898,  5.52480991,
5.5524045,   1.98648086,  4.65813625,  5.73834787,  5.2805168,   2.35590066,
4.36276439,  5.66855415, 15.71212956,  5.58639466, -0.03465203,  3.92718468,
0.27985268,  6.10848261,  1.01198014,  5.48098847,  0.46659923,  2.86042039,
1.29931791,  5.49722482,  4.44790528,  2.51640997]
params_better = np.array(params_better_list)
(E1, filling_gs, sweet_spot) = fermi_hubbard_half_filling_more_entanglement(1, 8, 0, 2, 2, 100, 20, True, params_better)
print(E1, filling_gs)
E0 = E1 - 1
iter = 1
while abs(E0-E1) > 1e-5:
    E0 = E1
    params_better = sweet_spot
    (E1, filling_gs, sweet_spot) = fermi_hubbard_half_filling_more_entanglement(1, 8, 0, 2, 2, 100, 20, True, params_better)
    iter = iter + 1
    print(E1, filling_gs)
print(sweet_spot)
print(iter)