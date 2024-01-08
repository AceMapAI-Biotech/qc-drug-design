

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
    # 计算C和S之间的距离
    dist_vector = positions_now[atom_1] - positions_now[atom_2]
    # dist_vector = positions_now[191] - positions_now[2673]
    # dist_vector = positions_now[191] - positions_now[2673]
    dist_value = dist_vector.value_in_unit(unit.angstroms)
    # dist_value = np.sqrt(np.sum(dist_vector**2)) * unit.angstroms
    # 计算欧氏距离
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

def update_qm_energy_and_forces(step, integrator, simulation, system):
    # 从上下文中获取当前坐标和力
    state = simulation.context.getState(getPositions=True, getForces=True)
    coordinates = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    #获取mm的电荷与charge
    mm_atom_charges = [system.getParticleMass(index).value_in_unit(unit.dalton) for index in mm_atoms]
    mm_atom_coords = [coordinates[i] for i in mm_atoms]
    mm_atom_coords_angstrom = np.array(mm_atom_coords) * unit.angstrom
    mm_atom_coords_bohr = mm_atom_coords_angstrom.value_in_unit(unit.bohr)

    # 从pdb.topology.atoms()提取原子名称
    atom_names = [atom.element.symbol for atom in prmtop.topology.atoms()]

    # 仅选取位于QM区域的原子坐标 name已经定义为全局变量
    qm_atom_coords = [coordinates[i] for i in qm_atoms]
    qm_atom_names = [atom_names[i] for i in qm_atoms]

    # 使用选定的QM原子构建一个新的pyscf分子对象
    mol = pyscf.gto.Mole()
    mol.atom = '\n'.join([f'{atom_name} {" ".join(map(str, coord))}' for
                          atom_name, coord in zip(qm_atom_names, qm_atom_coords)])
    mol.basis = '6-31g'

    # 设置分子的旋转数（示例：0）
    # mol.spin = 0  # 假设系统 未配对电子书为0  计算公式为 2S = N_alpha - N_beta，其中 N_alpha 和 N_beta 分别表示 α 和 β 自旋电子的数量。
    mol.spin = 0
    # 必须考虑全部电子 不能只是最外层电子
    #atom_num_electrons = {'N': 7, 'C': 6, 'O': 8,'H':1,'S':16} #TODO 把所有QM区域的原子都考虑进去
    #num_electrons = sum([atom_num_electrons[atom_name] for atom_name in qm_atom_names])
    #mol.nelectron = num_electrons  # 设置正确的电子数目，需要确保与分子结构匹配

    # 从模拟对象中获取系统和qmmm_force
    system = simulation.system
    qmmm_force = system.getForce(system.getNumForces() - 1)  # 获取最后一个添加的力（qmmm_force）

    #初始化mol
    mol.build()
    # 创建一个RHF计算对象，将其与QMMM耦合，并将MM区域的原子电荷传递给QM计算
    # calc = pyscf.scf.RHF(mol)
    # 为 QM 区域指定一个密度拟合方法，以提高计算效率
    calc = pyscf.scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit')


    # 考虑MM区域影响的HF计算，并应用QMMM耦合
    #确认 注意这里的第二个参数是  mm的电荷和区域坐标
    # 创建一个 pyscf 的 HF 对象，用于 QM 区域的计算，并添加 MM 区域的电荷作为背景电荷 所以到底 添加什么的背景 和坐标 需要考虑
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
        # 调用pyscf计算QM区域的能量和力
        E_qm = hf.kernel()
        forces_qm = -hf.nuc_grad_method().kernel()

    # 更新自定义力中QM区域的能量和力
    qmmm_force.setGlobalParameterDefaultValue(0, E_qm)
    for i, force in enumerate(forces_qm):
        force_in_kJ_per_mol = [f*unit.hartree/unit.bohr for f in force] # 将力从哈特里/埃转换为kJ/mol * A
        qmmm_force.setParticleParameters(i, qm_atoms[i], force_in_kJ_per_mol)

    return E_qm

# 设置日志记录器
logger.add("logfile_{time}.log")

prmtop = app.AmberPrmtopFile('6oim_CYX_amber_success.prmtop')
inpcrd = app.AmberInpcrdFile('6oim_CYX_amber_success.inpcrd')
RST_FILE = 'prod_1.rst' # starting info


temperature = 300*unit.kelvin
friction = 1.0/unit.picosecond
pressure = 1.0*unit.atmospheres
barostatInterval = 25

system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1*unit.nanometer,constraints=app.HBonds)
system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostatInterval))

logger.info('setting of qm/mm field')
# 定义 QM 区域和 MM 区域的原子索引
#序号需要减1 ， python 从0开始
qm_atoms = [189,190,191,257,192]

atom_names = [atom.element.symbol for atom in prmtop.topology.atoms()]
mm_atoms = list(set(range(system.getNumParticles())) - set(qm_atoms)) # 剩余的原子作为 MM 区域
charges_mm = [system.getParticleMass(index).value_in_unit(unit.dalton) for index in mm_atoms]


# 创建一个 QM/MM 力，它将 QM 区域和 MM 区域连接起来，并计算 QM 区域的能量和力
qmmm_force = mm.CustomExternalForce('E_qm')
qmmm_force.addGlobalParameter('E_qm', 0.0)

for i in qm_atoms:
    qmmm_force.addParticle(i)

system.addForce(qmmm_force)

logger.info("Setting up the integrator...")
# 创建一个自定义积分器，它可以在每个时间步之前调用 pyscf 来更新 QM 区域的能量和力
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA  # Boltzmann 常数
temp = 300 * unit.kelvin  # 温度
timestep = 2 * unit.femtoseconds
friction = 468 / unit.picosecond

integrator = mm.CustomIntegrator(timestep)
integrator.addUpdateContextState()  # 更新上下文状态，例如温度和压力
integrator.addGlobalVariable("kT", kB.value_in_unit_system(unit.md_unit_system) * temp.value_in_unit(unit.kelvin))  # 将 kT 定义为全局变量
integrator.addGlobalVariable("friction", friction.value_in_unit(unit.picosecond**-1)) #阻尼项

# integrator.addComputePerDof("v", "v+0.5*dt*f/m")  # 使用速度 阻尼算法 算法更新速度
integrator.addComputePerDof("v", "v + 0.5*dt*f/m - dt*friction*v + sqrt(2*friction*kT*dt)*gaussian/m") #考虑阻尼项
# 已确认  需要确认x的更新方法  openmm 的自定义积分器是在更新速度后更新位置，因此是半程位置
# 不需要用 x+dt*v+0.5*dt*dt*f/m
integrator.addComputePerDof("x", "x+dt*v")  # 使用速度 Verlet 算法更新位置
integrator.addConstrainPositions()  # 限制键长和键角

integrator.addConstrainVelocities()  # 限制速度
logger.info("Integrator set up.\n")

logger.info("Creating the simulation object...")
# 创建一个模拟对象
# 使用CUDA平台进行GPU加速
try:
    platform = mm.Platform.getPlatformByName('CUDA')

    # 指定要使用的GPU，例如使用第0号和第1号GPU
    gpu_indices = '0,1'
    properties = {'CudaPrecision': 'mixed', 'CudaDeviceIndex': gpu_indices}
    # 创建一个模拟对象并指定CUDA平台和属性
    simulation = app.Simulation(prmtop.topology, system, integrator, platform, properties)
except:
    simulation = app.Simulation(prmtop.topology, system, integrator)

simulation.context.setPositions(inpcrd.positions) # 设置初始位置

if inpcrd.boxVectors is not None:
    simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)

with open(RST_FILE, 'r') as f:
    simulation.context.setState(XmlSerializer.deserialize(f.read()))

logger.info("Simulation object created.\n")


logger.info("Running production simulation...")
nsteps_prod = 60000
nlog_prod = 50

# 每 10 步输出一个 PDB 文件 画图的时候 enforcePeriodicBox 一定要设为false 不然小分子特容易跑偏
# simulation.reporters.append(app.PDBReporter('report_6oim_CYX12_mm.pdb',2,
#                                             enforcePeriodicBox=False))



simulation.reporters.append(app.StateDataReporter('report_6oim_CYX12_qmmm_report.csv',nlog_prod,
                                                  step=True, potentialEnergy=True, temperature=True,
                                                  kineticEnergy= True, totalEnergy=True,
                                                  volume = True)) # 每 10 步输出一个 CSV 文件，包含步数、势能和温度


dcd = DCDReporter('prod.dcd', nlog_prod)
dcd._dcd = DCDFile(dcd._out, simulation.topology, simulation.integrator.getStepSize())

simulation.reporters.append(dcd)
simulation.reporters.append(StateDataReporter(stdout, nlog_prod, step=True, speed=True, separator='\t\t'))
simulation.reporters.append(StateDataReporter('prod.log', nlog_prod, step=True, kineticEnergy=True, potentialEnergy=True, totalEnergy=True, temperature=True, volume=True, speed=True))

# 运行模拟
# simulation.step(nsteps)
logger.info("Starting run  production simulation step...")
with open(f'e_qm.log', 'w') as e_qm_file:
    for step in tqdm(range(nsteps_prod)):
        if step % nlog_prod == 0:
            draw_simulation_pdb(simulation,f'report_6oim_CYX12_qmmm_{str(step // nlog_prod)}.pdb')
        e_qm = update_qm_energy_and_forces(step, integrator,simulation,system)
        simulation.step(1)
        e_qm_file.write(f"{step}\t{e_qm}\n")

simulation.reporters.clear() # remove all reporters so the next iteration don't trigger them.

# Writing last frame information of stride
state = simulation.context.getState( getPositions=True, getVelocities=True )
with open('prod.rst', 'w') as f:
    f.write(XmlSerializer.serialize(state))

positions = simulation.context.getState(getPositions=True).getPositions()
PDBFile.writeFile(simulation.topology, positions, open('prod.pdb', 'w'))
logger.info("Production simulation finished.\n")

logger.info("All done!")


