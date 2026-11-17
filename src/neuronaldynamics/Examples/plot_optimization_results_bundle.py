import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

data_path = '/home/erik/data/diw_results/mpi_run5_updated_FPE_diff_coeff/'

parameters = ['theta', 'intensity', 'fraction_nmda', 'fraction_gaba_a', 'fraction_ex', 'pdf_offset', 'pdf_sigma',
             'voltage_sigma']

idx_list_PA = [1, 2, 4, 6]
idx_list_LM = [8, 9]

array_PA = np.zeros((len(idx_list_PA), len(parameters)))
array_LM = np.zeros((len(idx_list_PA), len(parameters)))
for n, idx in enumerate(idx_list_PA):
    fname = f'paper_iwave_opt_hierearch_no_{idx}_params.csv'
    full_fname = os.path.join(data_path, fname)
    array_PA[n] = np.loadtxt(full_fname, delimiter=',')
    array_PA[n][2:4] *=100
    array_PA[n][4:] *= 10

print(f'intensities: {array_PA[:, 1]}')
print(f'theta: {array_PA[:, 0]}')
plt.boxplot(array_PA, tick_labels=parameters)
plt.tight_layout()
plt.show()

# fixed_theta
data_path = '/home/erik/data/diw_results/mpi_run6_fixed_theta/'
idx_list_PA = [1, 2, 3, 4, 6]
idx_list_LM = [8, 9]
array_PA = np.zeros((len(idx_list_PA), len(parameters)-1))


for n, idx in enumerate(idx_list_PA):
    fname = f'paper_iwave_opt_hierarch_theta_fixed_no_{idx}_params.csv'
    full_fname = os.path.join(data_path, fname)
    array_PA[n] = np.loadtxt(full_fname, delimiter=',')
    array_PA[n][2:4] *=100
    array_PA[n][4:] *= 10

print(f'intensities: {array_PA[:, 0]}')

plt.boxplot(array_PA, tick_labels=parameters[1:])
plt.tight_layout()
plt.show()