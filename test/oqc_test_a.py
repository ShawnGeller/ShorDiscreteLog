from qiskit_aer import Aer
from shor.mosca_ekert.mosca_ekert import DiscreteLogMoscaEkertSeperateRegister

simulator = Aer.get_backend('aer_simulator')

me_algo = DiscreteLogMoscaEkertSeperateRegister(b=4, g=2, p=23, r=11, n=5, full_run=True, quantum_instance=simulator)

nshots=3
for k in range(10):
    result = me_algo.run(shots=nshots)
    print(result)
    if result['num_success'] >= 1:
        break

print("Found after ", (k+1)*nshots, "tries.")
print("m=", result['m'])