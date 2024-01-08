

from openmm import app
import openmm as mm
from openmm import unit
import pyscf
import pyscf.qmmm
import numpy as np
import mdtraj as md
from loguru import logger
from tqdm import tqdm
from openmm import *
from openmm.app import *
from sys import stdout, exit, stderr



def calc_atom_dist(simulation,atom_1,atom_2):
    state_now = simulation.context.getState(getPositions=True)
    positions_now = state_now.getPositions()
    # Calculate the distance between C and S.
    dist_vector = positions_now[atom_1] - positions_now[atom_2]
    # dist_vector = positions_now[191] - positions_now[2673]
    # dist_vector = positions_now[191] - positions_now[2673]
    dist_value = dist_vector.value_in_unit(unit.angstroms)
    # Compute the Euclidean distance
    dist = np.linalg.norm(dist_value)
    return dist




def getBondName(simulation,system,atom_index):
    # Iterate over all bonds
    for bond in simulation.topology.bonds():
        # Get the two atoms involved in the bond
        atom1, atom2 = bond
        # Print their names and elements
        # Check if this is the bond you want to inspect
        if (atom1.index == atom_index or atom2.index==atom_index): # assuming these are the indices of the atoms you want
        # if (atom1.index == 2709 or atom1.index==189):
            # Print the bond type
            # print(bond.type)
            print(f'{atom1.index} {atom1.name} {atom1.element} ---- {atom2.index} {atom2.name} {atom2.element}')

    print('system force num is',system.getNumForces())
    nforces = system.getNumForces()
    for i in range(nforces):
        force = system.getForce(i)
        if isinstance(force, mm.HarmonicBondForce):
            for j in range(force.getNumBonds()):
                # Get the parameters of each bond
                atom1,atom2, length, k = force.getBondParameters(j)
                if (atom1 == atom_index or atom2 ==atom_index):
                    print(f"Bond between particles{atom1},{atom2} has length {length} and force constant {k}. force type is {type(force)}")

def draw_simulation_pdb(simulation,file_name):
    stateForPDB = simulation.context.getState(getPositions=True)
    pdb_topology = md.Topology.from_openmm(simulation.topology)
    pdb_topology.residue(12).name = 'CYX'
    pdb_positions = stateForPDB.getPositions(asNumpy = True).value_in_unit(unit.nanometers)
    traj = md.Trajectory(pdb_positions,pdb_topology)
    traj.save(file_name)


# Define a Python function that can call pyscf to update the energy and force of the QM region
def update_qm_energy_and_forces(step, integrator, simulation, system):
    # Retrieve the current coordinates and forces from the context
    state = simulation.context.getState(getPositions=True, getForces=True)
    coordinates = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    #Get the charges and charge of mm
    mm_atom_charges = [system.getParticleMass(index).value_in_unit(unit.dalton) for index in mm_atoms]
    mm_atom_coords = [coordinates[i] for i in mm_atoms]
    mm_atom_coords_angstrom = np.array(mm_atom_coords) * unit.angstrom
    mm_atom_coords_bohr = mm_atom_coords_angstrom.value_in_unit(unit.bohr)

    # 从pdb.topology.atoms() get the atom name
    atom_names = [atom.element.symbol for atom in prmtop.topology.atoms()]

    # Only select the atomic coordinates located in the QM region.
    # Atom name has already been defined as a global variable
    qm_atom_coords = [coordinates[i] for i in qm_atoms]
    qm_atom_names = [atom_names[i] for i in qm_atoms]

    # Use the selected QM atoms to construct a new pyscf molecule object
    mol = pyscf.gto.Mole()
    mol.atom = '\n'.join([f'{atom_name} {" ".join(map(str, coord))}' for
                          atom_name, coord in zip(qm_atom_names, qm_atom_coords)])
    mol.basis = '6-31g'

    # Set the number of rotations of the molecule
    # Assume that the number of unpaired electrons in the system is 0. 
    # The calculation formula is 2S = N_alpha - N_beta, where N_alpha and N_beta represent the number of α and β spin electrons, respectively.
    mol.spin = 0
    # Set the correct number of electrons, ensuring it matches the molecular structure

    # Retrieve the system and qmmm_force from the simulation object
    system = simulation.system
    qmmm_force = system.getForce(system.getNumForces() - 1)  # 获取最后一个添加的力（qmmm_force）

    #Initialize mol
    mol.build()
    # Create an RHF computation object, couple it with QMMM, 
    # and pass the atomic charges of the MM region to the QM calculation
    # Specify a density fitting method for the QM region to improve computational efficiency
    calc = pyscf.scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')


    # Perform an HF calculation considering the effects of the MM region and apply QMMM coupling
    # Note that the second parameter here is the charge and regional coordinates of mm
    calc = pyscf.qmmm.mm_charge(calc, mm_atom_coords_bohr, mm_atom_charges)
    hf = calc

    do_qc = False
    if do_qc:
        from pyscf.mcscf import CASCI
        from tencirchem import HEA
        from tencirchem.molecule import n2
         # normal PySCF workflow
        hf = mol.HF()
        print(hf.kernel())
        casci = CASCI(hf, 2, 2)
         # set the FCI solver for CASSCF to be HEA
        casci.canonicalization = False
        casci.fcisolver = HEA.as_pyscf_solver()
        print(casci.kernel()[0])
        nuc_grad = casci.nuc_grad_method().kernel()
        print(nuc_grad)
        forces_qm = nuc_grad
    else:
        #  Create a pyscf HF object for calculations in the QM region and 
        #  add the charges of the MM region as background charges
        E_qm = hf.kernel()
        forces_qm = -hf.nuc_grad_method().kernel()

    # update the engery and charge of QM region to MM system
    qmmm_force.setGlobalParameterDefaultValue(0, E_qm)
    for i, force in enumerate(forces_qm):
        force_in_kJ_per_mol = [f*unit.hartree/unit.bohr for f in force] # 将力从哈特里/埃转换为kJ/mol * A
        qmmm_force.setParticleParameters(i, qm_atoms[i], force_in_kJ_per_mol)
    
    return E_qm


# set logger
logger.add("logfile_{time}.log")


prmtop = app.AmberPrmtopFile('./data/6oim_CYX_amber_success.prmtop')
inpcrd = app.AmberInpcrdFile('./data/6oim_CYX_amber_success.inpcrd')

system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1*unit.nanometer,constraints=app.HBonds)

#set mm parameters
dt = 0.004*unit.picoseconds
temperature = 300*unit.kelvin
friction = 1.0/unit.picosecond
pressure = 1.0*unit.atmospheres
barostatInterval = 25
system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostatInterval))




logger.info('setting of qm/mm field')
# Define the atomic indices for the QM region and MM region. 
# The index numbers need to be decremented by 1, as Python starts from 0. 
qm_atoms = [189,190,191,257,192]

atom_names = [atom.element.symbol for atom in prmtop.topology.atoms()]
mm_atoms = list(set(range(system.getNumParticles())) - set(qm_atoms)) # set the rest of the atoms as mm region
charges_mm = [system.getParticleMass(index).value_in_unit(unit.dalton) for index in mm_atoms]


# Create a QM/MM force that connects the QM region and MM region together, and calculates the energy and force of the QM region.
qmmm_force = mm.CustomExternalForce('E_qm')
qmmm_force.addGlobalParameter('E_qm', 0.0)

for i in qm_atoms:
    qmmm_force.addParticle(i)

system.addForce(qmmm_force)




logger.info("Setting up the integrator...")
# Create a custom integrator that can call pyscf before each time step to 
# update the energy and force of the QM region.
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA  # Boltzmann 
temp = 300 * unit.kelvin  # Temp
timestep = 2 * unit.femtoseconds
friction = 468 / unit.picosecond

integrator = mm.CustomIntegrator(timestep)
integrator.addUpdateContextState()  # Update the context state, such as temperature and pressure
integrator.addGlobalVariable("kT", kB.value_in_unit_system(unit.md_unit_system) * temp.value_in_unit(unit.kelvin))  # The kT is defined as a global variable
integrator.addGlobalVariable("friction", friction.value_in_unit(unit.picosecond**-1)) #friction

integrator.addComputePerDof("v", "v + 0.5*dt*f/m - dt*friction*v + sqrt(2*friction*kT*dt)*gaussian/m") #Considering the friction term 
# he custom integrator of openmm updates the position after updating the velocity, hence it's a mid-step position
integrator.addComputePerDof("x", "x+dt*v")  # Use the velocity Verlet algorithm to update the position
integrator.addConstrainPositions() 

integrator.addConstrainVelocities() 
logger.info("Integrator set up.\n")




logger.info("Creating the simulation object...")
# Create a simulation object 
# using gpu with CUDA
try:
    platform = mm.Platform.getPlatformByName('CUDA')

    # using 0 to 5th GPU
    gpu_indices = '0,1,2,3,4,5'
    properties = {'CudaPrecision': 'mixed', 'CudaDeviceIndex': gpu_indices}
    simulation = app.Simulation(prmtop.topology, system, integrator, platform, properties)
except:
    simulation = app.Simulation(prmtop.topology, system, integrator)
simulation.context.setPositions(inpcrd.positions) # set initial position
if inpcrd.boxVectors is not None:
    simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)


logger.info("Simulation object created.\n")





#Check if the peptide bond of the following amino acid is normal
getBondName(simulation,system,262)

#Check if the peptide bond of the previous amino acid is normal
getBondName(simulation,system,180)
#Calculate the distance between C-S atoms.
calc_atom_dist(simulation,189,190)
logger.info("Minimizing energy...")
simulation.minimizeEnergy()

simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)

getBondName(simulation,system,262)

getBondName(simulation,system,180)
calc_atom_dist(simulation,189,190)
logger.info("Energy minimized.\n")





logger.info("Running NVT pre-equilibration simulation...")
pre_equil_steps = 2000  #Preheating steps
simulation.step(pre_equil_steps)
logger.info("Finished NVT pre-equilibration simulation.\n")


nsteps = 2000
nlog_prod = 50
nruns = 30
current_step = 0

logger.info("setting  report...")
simulation.reporters.append(
    md.reporters.XTCReporter(file=str("./report_6oim_CYX_qmmmm.xtc"), reportInterval=nlog_prod)
)


# Output a CSV file every 10 steps, containing step number, potential energy, and temperature.
simulation.reporters.append(app.StateDataReporter('report_6oim_CYX12_qmmm_report.csv',nlog_prod,
                                                  step=True, potentialEnergy=True, temperature=True,
                                                  kineticEnergy= True, totalEnergy=True,
                                                  volume = True)) 


logger.info("Running production simulation...")

dcd = DCDReporter('prod.dcd', nlog_prod)
dcd._dcd = DCDFile(dcd._out, simulation.topology, simulation.integrator.getStepSize())

simulation.reporters.append(dcd)
simulation.reporters.append(StateDataReporter(stdout, nlog_prod, step=True, speed=True, separator='\t\t'))
simulation.reporters.append(StateDataReporter('prod.log', nlog_prod, step=True, kineticEnergy=True, potentialEnergy=True, totalEnergy=True, temperature=True, volume=True, speed=True))

for run in range(nruns):
    logger.info(f"Starting run {run+1} of {nruns}...")
    with open(f'e_qm_{run+1}.log', 'w') as e_qm_file:
        for step in tqdm(range(current_step, current_step+nsteps)):
            if step % nlog_prod == 0:
                draw_simulation_pdb(simulation,f'report_6oim_CYX12_qmmm_{str(step // nlog_prod)}.pdb')
            e_qm = update_qm_energy_and_forces(step, integrator,simulation,system)
            simulation.step(1)
            e_qm_file.write(f"{step}\t{e_qm}\n")

    # Writing last frame information of stride
    state = simulation.context.getState( getPositions=True, getVelocities=True )
    with open(f'prod_{run+1}.rst', 'w') as f:
        f.write(XmlSerializer.serialize(state))

    positions = simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeFile(simulation.topology, positions, open(f'prod_{run+1}.pdb', 'w'))

    logger.info(f"Run {run+1} of {nruns} finished.\n")
    current_step += nsteps

logger.info("All runs finished!")