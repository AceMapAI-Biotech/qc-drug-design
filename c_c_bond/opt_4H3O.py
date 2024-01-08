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

# Conformational optimization for 4H3O+.gjf molecule.

from pyscf import gto, scf, dft, lib, solvent
from pyscf.solvent import ddCOSMO
from pyscf.hessian import rhf
import numpy as np

# 4H3O+.gjf
mol = gto.M(
    atom = """
  O                  3.21366300    2.54589300    6.03383300
  C                  1.52829000   -0.87103600    8.02039200
  O                  0.71840800   -1.45566700    5.87838400
  C                  1.39046300   -0.66257300    6.54523500
  O                  1.94407800   -2.19841700    8.32927500
  H                  1.30438200   -2.84117000    7.93592000
  C                  1.91903100    0.54232700    5.91627200
  C                  1.58520500    0.82020200    4.48342900
  H                  0.61384800    0.37554800    4.22644800
  H                  2.34637700    0.35869900    3.84794200
  C                  1.59180300    2.34196300    4.29253200
  H                  1.37225500    2.60408700    3.24916800
  H                  0.82041900    2.77517900    4.94947400
  C                  2.96388800    2.91143300    4.70577300
  C                  2.68494500    1.39579800    6.60863400
  C                  3.00204000    1.14044900    8.00279300
  C                  3.92562600    2.04822400    8.68086500
  H                  4.33734700    2.87098300    8.15278500
  C                  4.25926500    1.84397200    9.95153400
  H                  4.93467700    2.50246600   10.43580800
  C                  3.69773800    0.71200400   10.67736900
  H                  3.96581300    0.54601500   11.68990600
  C                  2.85085000   -0.11186100   10.06954200
  H                  2.44455000   -0.93589000   10.59981500
  C                  2.47834300    0.10578200    8.66803800
  C                  2.93874700    4.43650500    4.58269300
  H                  2.74264200    4.71476800    3.53832000
  H                  2.14640600    4.84173400    5.22746500
  H                  3.91196300    4.84249900    4.89540300
  C                  4.11853100    2.40119200    3.79882200
  H                  3.91998700    2.69985300    2.76020400
  H                  5.06355000    2.84855400    4.13437600
  H                  4.19960100    1.30860800    3.85557800
  C                  0.08121800   -0.80909000    8.64934900
  H                 -0.50551700   -1.61056500    8.18165900
  H                  0.18067500   -0.96243900    9.73241600
  C                 -0.61993700    0.51199400    8.35020300
  C                 -0.40497000    1.72232600    9.14762800
  H                  0.25942500    1.70195300    9.97419600
  C                 -1.04153300    2.85210100    8.82831500
  H                 -0.87475400    3.70954400    9.40433600
  C                 -1.97372100    2.88206600    7.66206000
  C                 -2.15678800    1.76493400    6.96667000
  H                 -2.81454600    1.76833200    6.13533700
  C                 -1.46660300    0.54601900    7.32326100
  H                 -1.62470100   -0.33147700    6.74946900
  O                 -2.70611100    4.02820600    7.20433600
  H                  0.09910814   -2.05869259    4.20078105
  H                 -0.06066335   -1.80839024    2.17864806
  H                 -0.90279985   -3.44014722    3.07607509
  O                 -0.28811835   -2.43574335    3.15183474
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
   O   3.044417   2.394898   6.209818
   C   1.604142  -1.018738   8.063376
   O   0.853895  -1.754139   5.988680
   C   1.513862  -0.787270   6.562698
   O   2.016691  -2.335146   8.306667
   H   1.361439  -2.940192   7.981512
   C   2.008317   0.304805   5.906380
   C   1.863003   0.583179   4.426360
   H   1.029878   0.050401   3.998452
   H   2.750172   0.241662   3.899247
   C   1.644201   2.079499   4.217210
   H   1.615362   2.313068   3.158290
   H   0.686474   2.365102   4.635844
   C   2.748535   2.913279   4.856354
   C   2.669779   1.270058   6.688091
   C   3.066595   1.033682   8.077462
   C   3.929099   1.921753   8.727106
   H   4.257674   2.808488   8.222355
   C   4.356849   1.649016  10.008216
   H   5.021837   2.326911  10.510760
   C   3.926478   0.487128  10.643163
   H   4.266589   0.264913  11.638539
   C   3.061920  -0.385348  10.007834
   H   2.735162  -1.279001  10.503180
   C   2.616275  -0.112429   8.720448
   C   2.305252   4.349159   5.088714
   H   2.098818   4.818910   4.132809
   H   1.401214   4.383130   5.682693
   H   3.083599   4.915943   5.587628
   C   4.073742   2.846757   4.107293
   H   3.956538   3.288385   3.123616
   H   4.837402   3.401579   4.640048
   H   4.420683   1.827395   3.983968
   C   0.162909  -0.769396   8.657759
   H  -0.399558  -1.675210   8.450852
   H   0.285247  -0.712498   9.734458
   C  -0.568319   0.412437   8.077506
   C  -0.228682   1.742243   8.379468
   H   0.435991   1.941212   9.206267
   C  -0.700304   2.805076   7.651757
   H  -0.430161   3.812776   7.918445
   C  -1.570760   2.632695   6.511808
   C  -2.025719   1.277726   6.325489
   H  -2.811636   1.116351   5.603581
   C  -1.524493   0.234152   7.074606
   H  -1.879987  -0.765992   6.872709
   O  -1.891147   3.567913   5.761833
   H   0.437452  -1.538175   5.148523
   H  -1.378053  -0.274480   4.417818
   H  -1.660166  -1.532056   3.590594
   O  -0.964114  -1.012558   3.972314
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
mol = gto.Mole()

mol.atom = """
 O                 -1.12823800    0.63904700   -1.60489900
 C                  2.11270000   -0.89702300    0.45396900
 O                  1.00849600   -2.95386800    0.68714000
 C                  0.91491200   -1.75718200    0.15204000
 O                  3.31093900   -1.65776800    0.50275900
 H                  3.21218000   -2.33931500    1.19065800
 C                 -0.22346600   -1.24948400   -0.45541700
 C                 -1.56131500   -1.94516600   -0.40581100
 H                 -1.66430300   -2.47005900    0.54871000
 H                 -1.63376100   -2.70098000   -1.19720400
 C                 -2.68824400   -0.92007800   -0.52915000
 H                 -3.64989300   -1.42677400   -0.65360500
 H                 -2.73706900   -0.32334800    0.38733400
 C                 -2.48696100    0.02096000   -1.70904400
 C                 -0.10878600    0.01309600   -1.06697100
 C                  1.16125700    0.71846300   -1.20660400
 C                  1.26982700    1.86113400   -2.01851600
 H                  0.39671000    2.23300400   -2.54156100
 C                  2.49452900    2.50625200   -2.14680100
 H                  2.57909800    3.38680400   -2.77625300
 C                  3.61551600    2.01704300   -1.46444200
 H                  4.57427100    2.51707600   -1.56619400
 C                  3.50897200    0.89045500   -0.64716000
 H                  4.37495900    0.52776000   -0.10499600
 C                  2.28602400    0.23103500   -0.52047500
 C                 -3.42976200    1.21149000   -1.65146400
 H                 -4.46075900    0.85701600   -1.75276100
 H                 -3.33280400    1.73545500   -0.69505200
 H                 -3.22150900    1.91111700   -2.46754900
 C                 -2.52222000   -0.67655700   -3.06425000
 H                 -3.51078800   -1.12450400   -3.21043200
 H                 -2.34337400    0.04428300   -3.86825900
 H                 -1.76908800   -1.46750100   -3.13036700
 C                  1.85291200   -0.26255500    1.90779700
 H                  1.83836400   -1.11609900    2.59495500
 H                  2.75876100    0.32092600    2.09826400
 C                  0.61285300    0.56386800    2.06226900
 C                  0.55900000    1.90976700    1.65709800
 H                  1.45994800    2.38380300    1.27194100
 C                 -0.62200600    2.64521000    1.71433900
 H                 -0.63295800    3.68161300    1.38190400
 C                 -1.83964200    2.07428900    2.19193700
 C                 -1.76013400    0.71655900    2.62868600
 H                 -2.66380200    0.24360100    3.00830500
 C                 -0.57509500   -0.00652500    2.55358300
 H                 -0.56747800   -1.04928400    2.86821900
 O                 -2.96623700    2.75028600    2.22689400
 H                  0.25476100   -3.60494700    0.51339200
 H                 -1.33284600   -4.84978000   -0.26150400
 H                 -0.05021500   -5.65853900    0.01084400
 O                 -0.63791400   -4.99040300    0.40529600
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

