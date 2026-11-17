"""
FPE_vs_IK_network.py with a sine input: FPE of Izhikevich populations (three adaptation closures) vs. IK networks with
20k, 50k and 100k neurons per population, 1 population and 3 populations with feedback to exc 1, five noise levels.

Input into exc 1: I0 + I0 sin(2 pi SINE_FREQUENCY t) for t < T_ON, then 0 (I0 = 150 pA, 100 Hz, T = 200 ms, T_ON =
100 ms, as the sine of FPE_Izhikevich_closure_test.py). All other settings (grid, coupling, noise levels, network sizes)
are those of FPE_vs_IK_network.py.

Outputs in Examples/solver_tests/FPE_vs_IK_network_sine (same files as FPE_vs_IK_network.py).

Usage (from src/neuronaldynamics):  python Examples/solver_tests/FPE_vs_IK_network_sine.py [--repeat [N]]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # FPE_vs_IK_network (same folder)
import FPE_vs_IK_network as net

if __name__ == '__main__':
    net.set_input('sine')
    net.main(net.parse_repeat())
