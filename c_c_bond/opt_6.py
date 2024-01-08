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

# Conformational optimization for 6.gjf molecule.

from pyscf import gto, scf, dft, lib, solvent
from pyscf.solvent import ddCOSMO
from pyscf.hessian import rhf
import numpy as np

# 6.gjf 
mol = gto.M(
    atom = """
    C                 -3.43826600   -2.08244300    0.02336700
    C                 -2.28089000   -2.89537600   -0.05080200
    C                 -1.02828100   -2.31706000   -0.10462100
    C                 -0.87541500   -0.90544700   -0.08893200
    C                 -2.04800200   -0.08409300   -0.00830200
    C                 -3.32455700   -0.70744800    0.04622000
    C                  0.40580800   -0.27762200   -0.15137900
    C                  0.54111500    1.10120500   -0.14413900
    C                 -0.63659200    1.88763500   -0.03177700
    C                 -1.89366600    1.32363500    0.02764500
    O                  1.48198800   -1.12925400   -0.24079200
    C                  2.80896200   -0.58720600    0.07160000
    C                  2.96659400    0.74489900   -0.66436800
    C                  1.89863300    1.75546800   -0.24055500
    C                  3.78386800   -1.62954200   -0.45789300
    C                  2.92817000   -0.44427400    1.58984600
    H                 -4.42082700   -2.54463800    0.06452700
    H                 -2.38038100   -3.97717300   -0.06399500
    H                 -0.14248700   -2.94022800   -0.15984100
    H                 -4.21233000   -0.08660000    0.10535300
    H                  3.96671000    1.14719700   -0.47460300
    H                  2.88580700    0.55018000   -1.74044600
    H                  1.87553900    2.57436600   -0.97139000
    H                  2.16347200    2.20877400    0.72508600
    H                  3.67824800   -1.74472700   -1.54220100
    H                  4.81144900   -1.32014200   -0.23947500
    H                  3.60716900   -2.60002700    0.01903900
    H                  2.82937300   -1.42423000    2.06897700
    H                  3.90861000   -0.02854800    1.84731800
    H                  2.15892100    0.21733400    1.99932000
    O                 -3.02867700    2.11293100    0.13232600
    H                 -2.75070300    3.04606200    0.15533100
    O                 -0.59635200    3.27098800    0.01980900
    H                  0.32088600    3.59110100    0.02078800
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
   C  -3.406279  -2.053850   0.018601
   C  -2.248938  -2.863895  -0.013194
   C  -1.010599  -2.294358  -0.054259
   C  -0.865968  -0.883956  -0.065361
   C  -2.020301  -0.079764  -0.033650
   C  -3.296547  -0.695065   0.009130
   C   0.419461  -0.258474  -0.115376
   C   0.541687   1.096205  -0.138754
   C  -0.641938   1.886575  -0.074291
   C  -1.876702   1.336333  -0.033342
   O   1.479318  -1.100057  -0.169595
   C   2.790062  -0.595536   0.077710
   C   2.953688   0.733254  -0.663086
   C   1.896936   1.753796  -0.244717
   C   3.731061  -1.650934  -0.488699
   C   2.992333  -0.456594   1.588198
   H  -4.377535  -2.514603   0.049893
   H  -2.348575  -3.934678  -0.005787
   H  -0.129397  -2.904847  -0.080583
   H  -4.170291  -0.072853   0.032749
   H   3.950984   1.127573  -0.494453
   H   2.858933   0.539212  -1.727391
   H   1.857816   2.558434  -0.971557
   H   2.172802   2.211877   0.704834
   H   3.566876  -1.776743  -1.552983
   H   4.765922  -1.363376  -0.329441
   H   3.563466  -2.606489  -0.004168
   H   2.883680  -1.423899   2.066150
   H   3.986652  -0.077533   1.805443
   H   2.268174   0.215050   2.033636
   O  -2.998780   2.092332   0.009437
   H  -2.754457   3.008836  -0.039255
   O  -0.566418   3.259282  -0.120279
   H  -0.003230   3.594444   0.565483
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
# from  energy cal 6.gjf 

mol = gto.Mole()

mol.atom = """
 C                 -3.43826600   -2.08244300    0.02336700
 C                 -2.28089000   -2.89537600   -0.05080200
 C                 -1.02828100   -2.31706000   -0.10462100
 C                 -0.87541500   -0.90544700   -0.08893200
 C                 -2.04800200   -0.08409300   -0.00830200
 C                 -3.32455700   -0.70744800    0.04622000
 C                  0.40580800   -0.27762200   -0.15137900
 C                  0.54111500    1.10120500   -0.14413900
 C                 -0.63659200    1.88763500   -0.03177700
 C                 -1.89366600    1.32363500    0.02764500
 O                  1.48198800   -1.12925400   -0.24079200
 C                  2.80896200   -0.58720600    0.07160000
 C                  2.96659400    0.74489900   -0.66436800
 C                  1.89863300    1.75546800   -0.24055500
 C                  3.78386800   -1.62954200   -0.45789300
 C                  2.92817000   -0.44427400    1.58984600
 H                 -4.42082700   -2.54463800    0.06452700
 H                 -2.38038100   -3.97717300   -0.06399500
 H                 -0.14248700   -2.94022800   -0.15984100
 H                 -4.21233000   -0.08660000    0.10535300
 H                  3.96671000    1.14719700   -0.47460300
 H                  2.88580700    0.55018000   -1.74044600
 H                  1.87553900    2.57436600   -0.97139000
 H                  2.16347200    2.20877400    0.72508600
 H                  3.67824800   -1.74472700   -1.54220100
 H                  4.81144900   -1.32014200   -0.23947500
 H                  3.60716900   -2.60002700    0.01903900
 H                  2.82937300   -1.42423000    2.06897700
 H                  3.90861000   -0.02854800    1.84731800
 H                  2.15892100    0.21733400    1.99932000
 O                 -3.02867700    2.11293100    0.13232600
 H                 -2.75070300    3.04606200    0.15533100
 O                 -0.59635200    3.27098800    0.01980900
 H                  0.32088600    3.59110100    0.02078800
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