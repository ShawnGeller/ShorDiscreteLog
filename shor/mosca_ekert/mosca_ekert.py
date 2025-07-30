from typing import Dict, Optional, Union, Callable

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
# from qiskit.utils import QuantumInstance
from qiskit.circuit.library import QFT
from qiskit.providers import Backend  #, BaseBackend

from shor.arithmetic.brg_mod_exp import mod_exp_brg

import math
import numpy as np
from abc import ABC, abstractmethod
from fractions import Fraction

from shor.arithmetic.brg_mod_mult import mult_mod_N_c


def mod_mul_brg(n, a, p):
    return mult_mod_N_c(a, p)


class DiscreteLogMoscaEkert():
    def __init__(self,
                 b: int,
                 g: int,
                 p: int,
                 r: int = -1,
                 n: int = -1,
                 mod_exp_constructor: Callable[[int, int, int], QuantumCircuit] = None,
                 full_run: bool = False,
                 quantum_instance: Optional[
                     # Union[QuantumInstance, Backend, BaseBackend, Backend]] = None) -> None:
                     Union[Backend]] = None) -> None:
        """
        Args:
            b: Finds discrete logarithm of b with respect to generator g and module p.
            g: Generator.
            p: Prime module.
            r: The order of g if it is known (otherwise it will be calculated)
            n: The size of the top register, if not given it will be inferred from the module p
            mod_exp_constructor: Function returns modular exponentiation circuit: (n, a, p) -> QuantumCircuit
                (n = size of top register)
            run_full (default: False), if set to True, the algorithm will not stop after finding the result,
                but will examine all results and log the probability
            quantum_instance: Quantum Instance or Backend
        """
        self._b = b
        self._g = g
        self._p = p
        self._r = r
        self._n = n
        self._full_run = full_run
        self._quantum_instance = quantum_instance
        self._me_circuit = None  # memoized transpiled circuit.

        if mod_exp_constructor is None:
            self._mod_exp_constructor = mod_exp_brg
        else:
            self._mod_exp_constructor = mod_exp_constructor

    @abstractmethod
    def _exec_qc(self, n, b, g, p, r, mod_exp_constructor, quantum_instance, shots) -> [(int, int, int)]:
        pass

    def get_circuit(self):
        if self._n == -1:
            n = math.ceil(math.log(self._p, 2))
        else:
            n = self._n
        print("constructing with size n=", n)
        return self._construct_circuit(n, self._b, self._g, self._p, self._r, self._mod_exp_constructor)

    @abstractmethod
    def _construct_circuit(self, n: int, b: int, g: int, p: int, r: int, mod_exp_constructor) -> QuantumCircuit:
        pass

    @staticmethod
    def find_period(result_list: [(int, int, int)], n: int, p: int, g: int) -> int:
        """The order of the generator is not given, it has to be determined.
        The list of results for the whole circuit will be used, since the
        measurement of stage1 allows for phase estimation of the operator
        (similar to Shor's algorithm for prime factorization)."""
        smallest_fitting_denominator = p + 1  # init to p+1 (sth larger than the real result)
        for (y1, _, _) in result_list:
            meas_div = y1 / (2 ** n)
            frac_meas = Fraction(meas_div).limit_denominator(p - 1)

            # check if denominator fits
            r_candidate = frac_meas.denominator
            if g ** r_candidate % p == 1:
                # fits
                if r_candidate < smallest_fitting_denominator:
                    smallest_fitting_denominator = r_candidate

        print("Found r=", smallest_fitting_denominator)
        return smallest_fitting_denominator

    # Manually reproduced since quantumalgorithm interface changed
    def run(self, shots:int = 100) -> Dict:
        return self._run(shots)

    def _run(self, shots:int) -> Dict:
        # construct circuit

        # size of top register is enough to hold all values mod p
        if self._n == -1:
            n = math.ceil(math.log(self._p, 2))
        else:
            n = self._n
        # print("constructing with size n=", n)
        # print("top register size: ", n)

        if self._quantum_instance is not None:
            # shots = 100

            result_list = self._exec_qc(n, self._b, self._g, self._p, self._r, self._mod_exp_constructor,
                                        self._quantum_instance, shots)

            num_fail = 0
            num_success = 0

            correct_m = -1
            if self._r == -1:
                # period has to be calculated
                self._r = self.find_period(result_list, n, self._p, self._g)

            # The following function is used to process the samples in the result_list.
            # It is only used here, so defined locally.
            def calcres(x, y, g, p, r, n):
                # x,y: The two sampled values in the quantum discrete log algorithm.
                # g: The base.
                # p: The modulus for Z_n.
                # r: g^r = 1 mod p.
                # n: 2^n is the range of the Fourier transform for sampling x,y.
                # For the return value, see the following calculation.
                k = int(round(x * r / (2 ** n)))
                l = int(round(y * r / (2 ** n)))
                # Find m such that l = mk mod r, if possible.
                if k == 0:  # This is no help.
                    return ((0,-1,k,l))  # Default return for m is 0.
                kdiv = math.gcd(k, r)
                if kdiv > 1: # l is a multiple of k mod r iff kdiv divides l.
                    ldiv = math.gcd(l, r)
                    if ldiv % kdiv == 0:
                        k = k // kdiv
                        l = l // kdiv
                        r = r // kdiv   # This gives the new relevant order for determining the multiple of k that is l.
                    else:
                        return ((0,k,l))
                dlog = (l * pow(k, -1, r)) % r
                return ((dlog, k, l))

            processed_result_list = [ (x[0],x[1]) + calcres(x[0],x[1],self._g,self._p,self._r,n) + (x[2],) for x in result_list]
            processed_result_list = [x for x in processed_result_list if pow(self._g,x[2],self._p) == self._b]  # Filter for correct discrete log.

            num_success = sum(x[-1] for x in processed_result_list)
            if num_success >= 1: correct_m = processed_result_list[0][2]

            # Revision by Manny:
            # The original version of this project used a poor order of rounding and modular operations
            # that resulted in poor efficiency scaling in terms of number of samples as n increased.


            return {"m": correct_m, "success_prob": num_success / shots, "num_success": num_success, "successes": sorted(processed_result_list,key = lambda x: x[0])}

        return {"m": -1, "success_prob": 0, "num_success": 0, "successes": None}


class DiscreteLogMoscaEkertSharedRegister(DiscreteLogMoscaEkert):
    def __init__(self,
                 b: int,
                 g: int,
                 p: int,
                 r: int = -1,
                 n: int = -1,
                 mod_exp_constructor: Callable[[int, int, int], QuantumCircuit] = None,
                 full_run: bool = False,
                 quantum_instance: Optional[
                     # Union[QuantumInstance, Backend, BaseBackend, Backend]] = None) -> None:
                     Union[Backend]] = None) -> None:
        super().__init__(b, g, p, r, n, mod_exp_constructor, full_run, quantum_instance)

    def _exec_qc(self, n, b, g, p, r, mod_exp_constructor, quantum_instance, shots) -> [(int, int, int)]:
        if self._me_circuit is None:
            self._me_circuit = transpile(self._construct_circuit(n, b, g, p, r, mod_exp_constructor),
                               quantum_instance)

        counts = quantum_instance.run(self._me_circuit, shots=shots).result().get_counts(self._me_circuit)

        res = list()
        for result in counts.keys():
            # split result measurements
            result_s = result.split(" ")
            m_stage1 = int(result_s[1], 2)
            m_stage2 = int(result_s[0], 2)

            res.append((m_stage1, m_stage2, counts[result]))

        return res

    def _construct_circuit(self, n: int, b: int, g: int, p: int, r: int, mod_exp_constructor) -> QuantumCircuit:
        # infer size of circuit from modular exponentiation circuit
        mod_exp_g = mod_exp_constructor(n, g, p)
        mod_exp_b = mod_exp_constructor(n, b, p)

        iqft = QFT(n).inverse()

        total_circuit_qubits = mod_exp_g.num_qubits
        bottom_register_qubits = total_circuit_qubits - n

        topreg = QuantumRegister(n, "top")
        botreg = QuantumRegister(bottom_register_qubits, "bot")
        meas_stage1 = ClassicalRegister(n, "m1")
        meas_stage2 = ClassicalRegister(n, "m2")

        me_circuit = QuantumCircuit(topreg, botreg, meas_stage1, meas_stage2)

        # H on top
        me_circuit.h(topreg)

        # 1 on bottom
        me_circuit.x(botreg[0])

        # mod exp g^x mod p
        me_circuit.append(mod_exp_g, me_circuit.qubits)

        # iqft top
        me_circuit.append(iqft, topreg)

        # measure top register (stage 1)
        me_circuit.measure(topreg, meas_stage1)

        # reset top register
        me_circuit.reset(topreg)

        # h on top again
        me_circuit.h(topreg)

        # mod exp b^x' mod p
        me_circuit.append(mod_exp_b, me_circuit.qubits)

        # iqft top
        me_circuit.append(iqft, topreg)

        # measurement stage 2
        me_circuit.measure(topreg, meas_stage2)

        #print(me_circuit)

        return me_circuit


class DiscreteLogMoscaEkertSeperateRegister(DiscreteLogMoscaEkert):
    def __init__(self,
                 b: int,
                 g: int,
                 p: int,
                 r: int = -1,
                 n: int = -1,
                 mod_exp_constructor: Callable[[int, int, int], QuantumCircuit] = None,
                 full_run: bool = False,
                 quantum_instance: Optional[
                     # Union[QuantumInstance, Backend, BaseBackend, Backend]] = None) -> None:
                     Union[Backend]] = None) -> None:
        super().__init__(b, g, p, r, n, mod_exp_constructor, full_run, quantum_instance)

    def _exec_qc(self, n, b, g, p, r, mod_exp_constructor, quantum_instance, shots) -> [(int, int, int)]:
        if self._me_circuit is None:
            self._me_circuit = transpile(self._construct_circuit(n, b, g, p, r, mod_exp_constructor),
                               quantum_instance)

        counts = quantum_instance.run(self._me_circuit, shots=shots).result().get_counts(self._me_circuit)

        res = list()
        for result in counts.keys():
            # split result measurements
            result_s = result.split(" ")
            m_stage1 = int(result_s[1], 2)
            m_stage2 = int(result_s[0], 2)

            res.append((m_stage1, m_stage2, counts[result]))

        return res

    def _construct_circuit(self, n: int, b: int, g: int, p: int, r: int, mod_exp_constructor) -> QuantumCircuit:
        # infer size of circuit from modular exponentiation circuit
        mod_exp_g = mod_exp_constructor(n, g, p)
        mod_exp_b = mod_exp_constructor(n, b, p)

        iqft = QFT(n).inverse()

        total_circuit_qubits = mod_exp_g.num_qubits
        bottom_register_qubits = total_circuit_qubits - n

        top1reg = QuantumRegister(n, "topstage1")
        top2reg = QuantumRegister(n, "topstage2")
        botreg = QuantumRegister(bottom_register_qubits, "bot")
        meas_stage1 = ClassicalRegister(n, "m1")
        meas_stage2 = ClassicalRegister(n, "m2")

        me_circuit = QuantumCircuit(top1reg, top2reg, botreg, meas_stage1, meas_stage2)

        # H on top
        me_circuit.h(top1reg)

        # 1 on bottom
        me_circuit.x(botreg[0])

        # mod exp g^x mod p
        me_circuit.append(mod_exp_g, list(top1reg) + list(botreg))

        # iqft top
        me_circuit.append(iqft, top1reg)

        # h on top2
        me_circuit.h(top2reg)

        # mod exp b^x' mod p
        me_circuit.append(mod_exp_b, list(top2reg) + list(botreg))

        # iqft top
        me_circuit.append(iqft, top2reg)

        # measure top register (stage 1)
        me_circuit.measure(top1reg, meas_stage1)

        # measurement stage 2
        me_circuit.measure(top2reg, meas_stage2)

        return me_circuit


class DiscreteLogMoscaEkertSemiClassicalQFT(DiscreteLogMoscaEkert):
    def __init__(self,
                 b: int,
                 g: int,
                 p: int,
                 r: int = -1,
                 n: int = -1,
                 mod_exp_constructor: Callable[[int, int, int], QuantumCircuit] = None,
                 full_run: bool = False,
                 quantum_instance: Optional[
                     # Union[QuantumInstance, Backend, BaseBackend, Backend]] = None) -> None:
                     Union[Backend]] = None) -> None:
        super().__init__(b, g, p, r, n, mod_exp_constructor, full_run, quantum_instance)

    def _exec_qc(self, n, b, g, p, r, mod_exp_constructor, quantum_instance, shots) -> [(int, int, int)]:
        if self._me_circuit is None:
            self._me_circuit = transpile(self._construct_circuit(n, b, g, p, r, mod_exp_constructor),
                                         quantum_instance)

        counts = quantum_instance.run(self._me_circuit, shots=shots).result().get_counts(self._me_circuit)

        res = list()
        for result in counts.keys():
            # split result measurements (is split bit by bit in this version)
            result_s = result.split(" ")

            #print("Result: ", result)

            # starts with stage 2
            m_stage2_bin = ""
            for i in range(0, n):
                m_stage2_bin = m_stage2_bin + result_s[i]

            m_stage1_bin = ""
            for i in range(0, n):
                m_stage1_bin = m_stage1_bin + result_s[n + i]

            #print(m_stage2_bin)
            #print(m_stage1_bin)

            m_stage1 = int(m_stage1_bin, 2)
            m_stage2 = int(m_stage2_bin, 2)

            res.append((m_stage1, m_stage2, counts[result]))

        return res

    @staticmethod
    def iqft_semi_classical(n: int, circ: QuantumCircuit, reg: QuantumRegister, cllist: [ClassicalRegister]):
        for i in reversed(range(0, n)):  # range is reversed as the first step of IQFT is a swap of the order
            swapped_i = n - i - 1  # index of this qubit after swap would have been performed

            # inverse rotations controlled by less significant qubits (classical now)
            # the rotations are performed with j's in the classical register - those are already implicitly swapped
            # therefore the swapped index is used here
            for j in range(0, swapped_i):
                # The register this qubit was measured in
                clreg = cllist[j]
                circ.p(-np.pi / (2 ** (swapped_i - j)), reg[i]).c_if(clreg, 1)

            circ.h(reg[i])

            # measure it in corresponding classical reg
            circ.measure(reg[i], cllist[swapped_i])

    def _construct_circuit(self, n: int, b: int, g: int, p: int, r: int, mod_exp_constructor) -> QuantumCircuit:
        # infer size of circuit from modular exponentiation circuit
        mod_exp_g = mod_exp_constructor(n, g, p)
        mod_exp_b = mod_exp_constructor(n, b, p)

        iqft = QFT(n).inverse()

        total_circuit_qubits = mod_exp_g.num_qubits
        bottom_register_qubits = total_circuit_qubits - n

        topreg = QuantumRegister(n, "top")
        botreg = QuantumRegister(bottom_register_qubits, "bot")

        cl_stage1 = []  # they have to be adressed as individual classical registers in c_if
        cl_stage2 = []

        me_circuit = QuantumCircuit(topreg, botreg)

        # init measurement registers as collection of single qubit registers
        for i in range(0, n):
            cl1 = ClassicalRegister(1, "f%d" % i)
            cl2 = ClassicalRegister(1, "s%d" % i)

            cl_stage1.append(cl1)
            cl_stage2.append(cl2)

        # Add measurement qubits per stage
        for i in range(0, n):
            me_circuit.add_register(cl_stage1[i])
        for i in range(0, n):
            me_circuit.add_register(cl_stage2[i])

        # First stage

        # H on top
        me_circuit.h(topreg)

        # 1 on bottom
        me_circuit.x(botreg[0])

        # mod exp g^x mod p
        me_circuit.append(mod_exp_g, list(topreg) + list(botreg))

        # semiclassical iqft top register into cl_stage1
        self.iqft_semi_classical(n, me_circuit, topreg, cl_stage1)

        # reset top
        me_circuit.reset(topreg)

        # second stage

        # h on top2
        me_circuit.h(topreg)

        # mod exp b^x' mod p
        me_circuit.append(mod_exp_b, list(topreg) + list(botreg))

        # semiclassical iqft top register into cl_stage2
        self.iqft_semi_classical(n, me_circuit, topreg, cl_stage2)

        return me_circuit


class DiscreteLogMoscaEkertOneQubitQFT(DiscreteLogMoscaEkert):
    def __init__(self,
                 b: int,
                 g: int,
                 p: int,
                 r: int = -1,
                 n: int = -1,
                 mod_mul_constructor: Callable[[int, int, int], QuantumCircuit] = None,
                 full_run: bool = False,
                 quantum_instance: Optional[
                     # Union[QuantumInstance, Backend, BaseBackend, Backend]] = None) -> None:
                     Union[Backend]] = None) -> None:
        """
        Needs direct access to the function construction a modular multiplication
        """
        super().__init__(b, g, p, r, n, mod_mul_constructor, full_run, quantum_instance)

        if mod_mul_constructor is None:
            self._mod_exp_constructor = mod_mul_brg
        else:
            self._mod_exp_constructor = mod_mul_constructor

    def _exec_qc(self, n, b, g, p, r, mod_mul_constructor, quantum_instance, shots) -> [(int, int, int)]:
        if self._me_circuit is None:
            self._me_circuit = transpile(self._construct_circuit(n, b, g, p, r, mod_mul_constructor),
                                         quantum_instance)

        counts = quantum_instance.run(self._me_circuit, shots=shots).result().get_counts(self._me_circuit)

        res = list()
        for result in counts.keys():
            # split result measurements (is split bit by bit in this version)
            result_s = result.split(" ")

            #print("Result: ", result)

            # starts with stage 2
            m_stage2_bin = ""
            for i in range(0, n):
                m_stage2_bin = m_stage2_bin + result_s[i]

            m_stage1_bin = ""
            for i in range(0, n):
                m_stage1_bin = m_stage1_bin + result_s[n + i]

            #print(m_stage2_bin)
            #print(m_stage1_bin)

            m_stage1 = int(m_stage1_bin, 2)
            m_stage2 = int(m_stage2_bin, 2)

            res.append((m_stage1, m_stage2, counts[result]))

        return res

    @staticmethod
    def one_qubit_qft_stage(n: int, a: int, p: int,
                            me_circuit: QuantumCircuit, topreg_single_qubit: QuantumRegister, botreg: QuantumRegister,
                            cllist: [ClassicalRegister], mod_mul_constructor):
        # act like there is a whole register while there actually is only this one qubit
        # this makes the code more similar to the other versions
        topreg = [topreg_single_qubit[0] for i in range(0, n)]

        for i in reversed(range(0, n)):
            # init control to H
            me_circuit.h(topreg[i])

            # exec gate
            mulgate = mod_mul_constructor(n, a ** (2 ** i) % p, p)  # U_g^(2^i)

            me_circuit.append(mulgate, [topreg[i], *botreg])

            swapped_i = n - i - 1  # index of this qubit after swap would have been performed
            # inverse rotations controlled by less significant qubits (classical now)

            # the rotations are performed with j's in the classical register - those are already implicitly swapped
            # therefore the swapped index is used here
            # The following is for an outdated way of handling classical control in qiskit:
            # for j in range(0, swapped_i):
            #     # The register this qubit was measured in
            #     clreg = cllist[j]
            #     me_circuit.p(-np.pi / (2 ** (swapped_i - j)), topreg[i]).c_if(clreg, 1)
            # Here is the updated way:
            for j in range(0, swapped_i):
                # The register this qubit was measured in
                clreg = cllist[j]
                with me_circuit.if_test((clreg, 1)):
                    me_circuit.p(-np.pi / (2 ** (swapped_i - j)), topreg[i])

            me_circuit.h(topreg[i])

            # measure it in corresponding classical reg
            me_circuit.measure(topreg[i], cllist[swapped_i])

            # reset to 0 so the rest of the algorithm can reuse this qubit
            me_circuit.reset(topreg[i])

            me_circuit.barrier()  # for visualisation purposes

    def _construct_circuit(self, n: int, b: int, g: int, p: int, r: int, mod_mul_constructor) -> QuantumCircuit:
        # infer size of circuit from modular exponentiation circuit
        mod_mul_dummy = mod_mul_constructor(n, g, p)

        iqft = QFT(n).inverse()

        total_circuit_qubits = mod_mul_dummy.num_qubits  #mod mul has botreg + 1 qubits already
        bottom_register_qubits = total_circuit_qubits - 1

        topreg = QuantumRegister(1, "top")
        botreg = QuantumRegister(bottom_register_qubits, "bot")

        cl_stage1 = []  # they have to be adressed as individual classical registers in c_if
        cl_stage2 = []

        me_circuit = QuantumCircuit(topreg, botreg)

        # init measurement registers as collection of single qubit registers
        for i in range(0, n):
            cl1 = ClassicalRegister(1, "f%d" % i)
            cl2 = ClassicalRegister(1, "s%d" % i)

            cl_stage1.append(cl1)
            cl_stage2.append(cl2)

        # Add measurement qubits per stage
        for i in range(0, n):
            me_circuit.add_register(cl_stage1[i])
        for i in range(0, n):
            me_circuit.add_register(cl_stage2[i])

        # First stage
        me_circuit.x(botreg[0])

        # performs stage1 = apply gate to superposition and do iqft in one qubit
        self.one_qubit_qft_stage(n, g, p, me_circuit, topreg, botreg, cl_stage1, mod_mul_constructor)

        # Second stage
        self.one_qubit_qft_stage(n, b, p, me_circuit, topreg, botreg, cl_stage2, mod_mul_constructor)

        return me_circuit
