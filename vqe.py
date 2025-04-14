import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

dev = qml.device('default.qubit', wires=4)

symbols = ["H", "H"]
geometry = np.array([[0., 0., -0.66140414], [0., 0., 0.66140414]])
molecule = qml.qchem.Molecule(symbols, geometry)
hamiltonian, qubits = qml.qchem.molecular_hamiltonian(molecule)

print("\nMolecular Hamiltonian:")
print(hamiltonian)
print(f"\nNumber of qubits needed: {qubits}")

@qml.qnode(dev)
def circuit(params):
    qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
    qml.DoubleExcitation(params, wires=[0, 1, 2, 3])
    return qml.expval(hamiltonian)

# Initialize parameters
params = np.array(0.0, requires_grad=True)
opt = AdamOptimizer(stepsize=0.4)

# Optimization loop
print("\nStarting VQE optimization...")
print("Step\tEnergy\t\tParameter")
print("---------------------------")

for step in range(100):
    params, energy = opt.step_and_cost(circuit, params)
    if step % 10 == 0:
        print(f"{step}\t{energy:.8f}\t{params:.8f}")

print("\nFinal results:")
print(f"Optimized parameter: {params:.8f}")
print(f"Ground state energy: {energy:.8f} Hartree")