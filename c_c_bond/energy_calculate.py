# This energy calculation process is a part of our QC pipeline which demonstrates how quantum computing 
# can be involved in the prodrug activation strategy. Here, we calculate the single-point energy of molecules during the reaction.
# This example describes the process of C=C bond cleavage.
# The main process involving the cleavage of the C=C bond concerns three molecules: 
# 4H3O+, 5, and 6; we now use the HF method to calculate the energies of these three molecules separately.
# 4H3O+  -->  5  +  6
# Cartesian coordinates are employed, with the molecular coordinates provided by Gaussian's .gjf input files.
# pyscf==2.3.0

# The energy calculation process is also for verification purposes and is based on two criteria.
# 1.The output energy is consistent with the Gaussian calculation results.
# 2.No imaginary frequencies appear in the frequency calculation.


from pyscf import gto, scf, dft, lib, solvent
from pyscf.solvent import ddCOSMO
from pyscf.hessian import rhf
import numpy as np

# 5.gjf
mol = gto.Mole()
mol.atom = """
 C                  0.00000800    1.39812400    0.00000000
 C                 -0.00000200    0.64420900    1.24608200
 C                 -0.00000200   -0.70980600    1.25056200
 C                  0.00003300   -1.46955300    0.00000000
 C                 -0.00000200   -0.70980600   -1.25056200
 C                 -0.00000200    0.64420900   -1.24608200
 O                 -0.00003200   -2.72480100    0.00000000
 C                  0.00001300    2.75406900    0.00000000
 H                 -0.00001100    1.20486000    2.17731400
 H                 -0.00001700   -1.27704300    2.17689300
 H                 -0.00001700   -1.27704300   -2.17689300
 H                 -0.00001100    1.20486000   -2.17731400
 H                  0.00001500    3.31704500    0.92947800
 H                  0.00001500    3.31704500   -0.92947800
"""

mol.basis = "6-311+g(d,p)"
mol.charge = 0
mol.spin = 0
mol.build()

# HF calculation
mf = scf.RHF(mol)
mf = scf.RHF(mol).run()

# COSMO calculation
with_solvent_ddcosmo = ddCOSMO(mf).run()

# Frequency analysis involving Hess calculation.
hess = rhf.Hessian(with_solvent_ddcosmo).kernel()

#########################
# Diagonalize the Hessian to get normal modes
eigval, eigvec = np.linalg.eigh(hess)

# Convert eigenvalues to vibrational frequencies (in cm^-1)
vib_freq = np.sqrt(np.abs(eigval)) * 219474.6  

print("Frequencies (cm^-1):")
print(vib_freq)

print("Normal Modes:")
print(eigvec)

# 6.gjf

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

# HF calculation
mf = scf.RHF(mol)
mf = scf.RHF(mol).run()

# COSMO calculation
with_solvent_ddcosmo = ddCOSMO(mf).run()

# Frequency analysis involving Hess calculation.
hess = rhf.Hessian(with_solvent_ddcosmo).kernel()

#########################
# Diagonalize the Hessian to get normal modes
eigval, eigvec = np.linalg.eigh(hess)

# Convert eigenvalues to vibrational frequencies (in cm^-1)
vib_freq = np.sqrt(np.abs(eigval)) * 219474.6  

print("Frequencies (cm^-1):")
print(vib_freq)

print("Normal Modes:")
print(eigvec)

# 4H3O+
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

# HF calculation
mf = scf.RHF(mol)
mf = scf.RHF(mol).run()



# COSMO calculation
with_solvent_ddcosmo = ddCOSMO(mf).run()

# Frequency analysis involving Hess calculation.
hess = rhf.Hessian(with_solvent_ddcosmo).kernel()

#########################
# Diagonalize the Hessian to get normal modes
eigval, eigvec = np.linalg.eigh(hess)

# Convert eigenvalues to vibrational frequencies (in cm^-1)
vib_freq = np.sqrt(np.abs(eigval)) * 219474.6  



print("Frequencies (cm^-1):")
print(vib_freq)

print("Normal Modes:")
print(eigvec)
