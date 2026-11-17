"""
FPE_Izhikevich_low_noise.py with a sine input: grid convergence of the Izhikevich FPE at low noise (1 population: all
closures, dv and dt study; 3 populations with feedback: conditional closure) against IK networks.

Input into exc 1: I0 + I0 sin(2 pi 100 Hz t) for t < 100 ms, then 0 (I0 = 150 pA, T = 200 ms), the sine input of
FPE_vs_IK_network.py. All other settings are those of FPE_Izhikevich_low_noise.py.

Outputs in Examples/solver_tests/FPE_Izhikevich_low_noise_sine (same files as FPE_Izhikevich_low_noise.py).

Usage (from src/neuronaldynamics):  python Examples/solver_tests/FPE_Izhikevich_low_noise_sine.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # FPE_Izhikevich_low_noise (same folder)
import FPE_Izhikevich_low_noise as low_noise

if __name__ == '__main__':
    low_noise.set_input('sine')
    low_noise.main()
