# This conformational optimization process is a part of our QC pipeline which demonstrates how quantum computing 
# can be involved in the prodrug activation strategy. Here, we calculate the optimal conformation of molecules during the reaction.
# This example describes the process of C=C bond cleavage.
# The main process involving the cleavage of the C=C bond concerns three molecules: 
# 4H3O+, 5, and 6; we now use the HF method to calculate the optimal conformation of these three molecules separately.
# In this calculation, the solvent model is chosen to be COSMO, and dispersion corrections are not considered.
# 4H3O+  -->  5  +  6
# Cartesian coordinates are employed, with the molecular coordinates provided by Gaussian's .gjf input files.
# pyscf==2.3.0

# The optimal conformation process is also for verification purposes and is based on two criteria.
# 1.The optimal conformations are consistent with the Gaussian calculation results.
# 2.No imaginary frequencies appear in the frequency calculation.

# Conformational optimization for 5.gjf molecule.

from pyscf import gto, scf, dft, lib, solvent
from pyscf.solvent import ddCOSMO
from pyscf.hessian import rhf
import numpy as np

# 5.gjf
mol = gto.M(
    atom = """
    C                  0.00017400    1.39788200    0.00000000
    C                 -0.00002900    0.64418600   -1.24604300
    C                 -0.00002900   -0.70982900   -1.25050500
    C                  0.00072100   -1.46902100    0.00000000
    C                 -0.00002900   -0.70982900    1.25050500
    C                 -0.00002900    0.64418600    1.24604300
    O                 -0.00073900   -2.72464600    0.00000000
    C                  0.00028800    2.75384400    0.00000000
    H                 -0.00021900    1.20486000   -2.17721200
    H                 -0.00035900   -1.27718700   -2.17669400
    H                 -0.00035900   -1.27718700    2.17669400
    H                 -0.00021900    1.20486000    2.17721200
    H                  0.00033300    3.31665200   -0.92953200
    H                  0.00033300    3.31665200    0.92953200
    """,
    basis='6-31+g(d)',
    charge=0,
    spin=0)

# Perform HF (Hartree-Fock) calculation.
mf = scf.RHF(mol)
mf = mf.density_fit() # Utilize density fitting for acceleration.
mf.with_solvent = solvent.DDCOSMO(mf) # Incorporate solvent effects, here using water as the default solvent.
mf.kernel()

# Conformational optimization
from pyscf.geomopt.berny_solver import optimize
conv_params = { # param
    'gradientmax': 1e-3,
    'gradientrms': 1e-4,
    'stepmax': 1e-3,
    'steprms': 1e-4,
}

mol_eq = optimize(mf, **conv_params)

# Based on the newly optimized conformation, perform an energy calculation on the molecule:
mol = gto.Mole()

mol.atom = """
 C   0.000091   1.404769  -0.000000
 C   0.000006   0.634201  -1.250651
 C  -0.000131  -0.696020  -1.259192
 C  -0.000208  -1.475784   0.000000
 C  -0.000131  -0.696020   1.259192
 C   0.000006   0.634201   1.250651
 O  -0.000322  -2.676784  -0.000000
 C   0.000226   2.737400   0.000000
 H   0.000070   1.187083  -2.173806
 H  -0.000181  -1.263378  -2.171542
 H  -0.000181  -1.263378   2.171542
 H   0.000070   1.187083   2.173806
 H   0.000261   3.301025  -0.915540
 H   0.000261   3.301025   0.915540
"""

mol.basis = "6-311+g(d,p)"
mol.charge = 0
mol.spin = 0
mol.build()

# Perform HF (Hartree-Fock) calculation.
mf = scf.RHF(mol)
mf = scf.RHF(mol).run()

# cosmo
with_solvent_ddcosmo = ddCOSMO(mf).run()

# Frequency analysis involving Hess calculation.
hess = rhf.Hessian(with_solvent_ddcosmo).kernel()

# diagonalize, the Hessian to get normal modes
eigval, eigvec = np.linalg.eigh(hess)

# convert eigenvalues to vibrational freq (in cm^-1)
vib_freq = np.sqrt(np.abs(eigval)) * 219474.6

print("Freq (cm^-1):")
print(vib_freq)


print("Normal Modes:")
print(eigvec)

# Perform energy calculations on the molecule based on the baseline conformation:
# from  energy cal 5.gjf 

mol = gto.Mole()

mol.atom = """
 C     0.00000800    1.39812400    0.00000000
 C    -0.00000200    0.64420900    1.24608200
 C    -0.00000200   -0.70980600    1.25056200
 C     0.00003300   -1.46955300    0.00000000
 C    -0.00000200   -0.70980600   -1.25056200
 C    -0.00000200    0.64420900   -1.24608200
 O    -0.00003200   -2.72480100    0.00000000
 C     0.00001300    2.75406900    0.00000000
 H    -0.00001100    1.20486000    2.17731400
 H    -0.00001700   -1.27704300    2.17689300
 H    -0.00001700   -1.27704300   -2.17689300
 H    -0.00001100    1.20486000   -2.17731400
 H     0.00001500    3.31704500    0.92947800
 H     0.00001500    3.31704500   -0.92947800
"""

mol.basis = "6-311+g(d,p)"
mol.charge = 0
mol.spin = 0
mol.build()

# Perform HF (Hartree-Fock) calculation.
mf = scf.RHF(mol)
mf = scf.RHF(mol).run()

# cosmo 
with_solvent_ddcosmo = ddCOSMO(mf).run()

# Frequency analysis involving Hess calculation.
hess = rhf.Hessian(with_solvent_ddcosmo).kernel()

# diagonalize, the Hessian to get normal modes
eigval, eigvec = np.linalg.eigh(hess)

# convert eigenvalues to vibrational freq (in cm^-1)
vib_freq = np.sqrt(np.abs(eigval)) * 219474.6

print("Freq (cm^-1):")
print(vib_freq)


print("Normal Modes:")
print(eigvec)
