"""
FPE_Izhikevich_low_noise.py with a 20 Hz sine input: I0 + I0 sin(2 pi 20 Hz t) into exc 1 for t < 100 ms, then 0
(I0 = 150 pA, T = 200 ms), see FPE_vs_IK_network_sine20Hz.py. All other settings are those of
FPE_Izhikevich_low_noise.py.

Outputs in Examples/solver_tests/FPE_Izhikevich_low_noise_sine_20Hz (same files as FPE_Izhikevich_low_noise.py).

Usage (from src/neuronaldynamics):  python Examples/solver_tests/FPE_Izhikevich_low_noise_sine20Hz.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # FPE_Izhikevich_low_noise (same folder)
import FPE_Izhikevich_low_noise as low_noise

if __name__ == '__main__':
    low_noise.set_input('sine', frequency=0.02,
                        out_dir=Path(__file__).resolve().parent / f'FPE_Izhikevich_low_noise_sine_20Hz{low_noise.net.method_suffix()}')
    low_noise.main()
