"""
FPE_vs_IK_network.py with a 20 Hz sine input: I0 + I0 sin(2 pi 20 Hz t) into exc 1 for t < 100 ms, then 0 (I0 = 150 pA,
T = 200 ms). The 100 Hz sine of FPE_vs_IK_network_sine.py is low-pass filtered by the populations (15-25 Hz rates) and
gives almost the step result, 20 Hz lies in the range the populations can follow. All other settings are those of
FPE_vs_IK_network.py.

Outputs in Examples/solver_tests/FPE_vs_IK_network_sine_20Hz<net.method_suffix()>, e.g. ..._20Hz_heun (same files as
FPE_vs_IK_network.py).

Usage (from src/neuronaldynamics):  python Examples/solver_tests/FPE_vs_IK_network_sine20Hz.py [--repeat [N]]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # FPE_vs_IK_network (same folder)
import FPE_vs_IK_network as net

if __name__ == '__main__':
    net.set_input('sine', frequency=0.02,
                  out_dir=Path(__file__).resolve().parent / f'FPE_vs_IK_network_sine_20Hz{net.method_suffix()}')
    net.main(net.parse_repeat())
