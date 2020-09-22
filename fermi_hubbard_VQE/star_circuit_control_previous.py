from star_circuit_previous import *
params_better_list = [ 6.17631912e+01,  3.92729658e+00,  2.09490917e+00,  3.64599246e+00,
1.21200580e+00,  3.96978984e+00,  4.01275718e+00,  4.30870989e+00,
6.31045828e+00,  2.35655265e+00,  3.30582640e+00,  1.76946990e+00,
1.81464603e+00,  6.02222535e+00,0.02,0.01,0.03,0.05,0.01,0.03,0.04,0.03,0.01,0.02,0.06,0.07,0.02,0.05,
8.11446902e-04,  5.49788958e+00,
-1.15828738e+00,  1.82734549e+00,  6.15077354e+00,  2.27540424e+00,
4.47702082e+00,  3.96929527e+00,  2.46744776e+00,  5.49861626e+00,
1.11699139e+00,  3.46195607e+00,  3.85245676e+00,  9.16442195e+00,
0.01,0.02,0.05,0.04,0.03,0.06,0.01,0.04,0.01,0.02,0.04,0.05,0.06,0.07]
params_better = np.array(params_better_list)
(E1, filling_gs, sweet_spot) = fermi_hubbard_half_filling_star_circuit(1,8,0,True,100,10, True, params_better)
#(E1, filling_gs, sweet_spot) = fermi_hubbard_half_filling_fully_connected(1,8,0,True,100,10, False, 1)
print(E1, filling_gs)
E0 = E1 - 1
iter = 1
while abs(E0-E1) > 1e-6:
    E0 = E1
    params_better = sweet_spot
    (E1, filling_gs, sweet_spot) = fermi_hubbard_half_filling_star_circuit(1,8,0,True,100,10, True, params_better)
    iter = iter + 1
    print(E1, filling_gs)
print(sweet_spot)
print(iter)