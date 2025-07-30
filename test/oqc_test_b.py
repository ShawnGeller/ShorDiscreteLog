from qiskit_aer import Aer
# from shor.arithmetic.hrs_mod_exp import mod_exp_hrs
from shor.mosca_ekert.mosca_ekert import DiscreteLogMoscaEkertOneQubitQFT

simulator = Aer.get_backend('aer_simulator')

me_algo = DiscreteLogMoscaEkertOneQubitQFT(b=9, g=2, p=227, r=226, n=8, full_run=True, quantum_instance=simulator)
# The following values can be used for testing:
# g=2: p=503, r=251, n=8; p=107, r=106, n=7; p=227, r=226, n=8; p=59, r=58, n=6.

nshots=10
for k in range(10):
    result = me_algo.run(shots=nshots)
    print(result)
    if result['num_success'] >= 1:
        break

print("Found after ", (k+1)*nshots, "tries.")
print("m=", result['m'])