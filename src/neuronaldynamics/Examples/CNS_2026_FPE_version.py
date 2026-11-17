import numpy as np
import matplotlib
from Model.DI_wave import DI_wave_simulation
matplotlib.use('TkAgg')

# uncomment for Times new roman font
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"]

plot_nykamp_basic = False
save_figs = True

dt = 0.01
dv = 0.01
T = 14

t = np.arange(0, T, dt)
Nt = t.shape[0]

fn_session = '/home/erik/Downloads/gpc.pkl'
# fn_session = 'C:\\Users\\emueller\\Downloads\\gpc.pkl'
# fn_session = 'C:\\Users\\User\\Downloads\\gpc.pkl'
# hdf5_path = "C:\\Users\\User\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data.hdf5"
# hdf5_path = "C:\\Users\\emueller\\Nextcloud\\TMS Neuro Projects\\M1_modeling\\DI_wave_data\\extracted_DI_waves\\DiLazarro_di_wave_data_detrended.hdf5"
hdf5_path = "/home/erik/Nextcloud_Uni/TMS Neuro Projects/M1_modeling/DI_wave_data/extracted_DI_waves/DiLazarro_di_wave_data.hdf5"


simulation_name = 'CNS_26_test_1'

measurement_dict_2020_120_PA_ch3 = dict(orientation='PA', threshold=120, year=2020, hdf5_path=hdf5_path, sigma=1.0)

measurement_dict = dict(orientation='PA', threshold=120, year=2013, hdf5_path=hdf5_path, sigma=1.0, channel=0)
parameters = {'intensity': 230, 'fraction_nmda': 0.5, 'fraction_gaba_a': 0.97, 'fraction_ex': 0.6, 'plot_align': False,
              'test_func_intensity': 2.0, 'test_func_t0': 0.35, 'enable_high_pass': True, 'min_delay': 0.0,
              'test_signal_from_file': True, 'i_scale': 5.148136e-9, 'detrend': False, 'plot_detrend': False,
              'fn_session': fn_session, 'T': T, 'name': simulation_name, 'dt': dt, 'mind_delay': 0,
              'theta': 30, 'delay_signal':False,
              'file_args': measurement_dict_2020_120_PA_ch3,
              'nykamp_parameters': {'connectivity_matrix': np.array([[0]]),
                                    'tau_ref': [0], #1.5
                                    'tau_mem': [12],
                                    'input_type': 'stochastic-current',
                                    'static_noise': True,
                                    'init_pdf_offset': 0,
                                    'init_pdf_sigma': 4,
                                    'init_pdf_weight': 0,
                                    'delay_kernel_type': 'alpha',
                                    'delay_kernel_parameters': {'n_alpha': 9, 'tau_alpha': 1/3},
                                    'dv': dv,
                                    'dt': dt,
                                    'solver': 'Hu-2021',
                                    'time_scheme': 'bdf2',
                                    'linear_solver': 'banded',
                                    'flux_scheme': 'bernoulli',
                                    'voltage_sigma': 5,
                                    'verbose': 1}}


di_model = DI_wave_simulation(parameters=parameters, logname=None)
di_model.get_test_signal(plot_d_wave_detection=True)
di_model.simulate()

di_model.mass_model.plot(heat_map=True, plot_input=True, plot_combined=True, z_limit=0.12, animate=True, savefig=save_figs)
di_model.labelsize=20
# di_model.plot_input_current(savefig=save_figs)
voltage_signal = di_model.mass_model_v_out

di_model.labelsize=17
di_model.plot_validation(fixed_ylim=False, save_fig=save_figs, labels=['Population Model', 'Measurement'])
di_model.mass_model.clean()
#
# parameters['T'] = 100
# lt_name = 'long_time_sim'
# parameters['name'] = lt_name
# di_model = DI_wave_simulation(parameters=parameters)
# import matplotlib.pyplot as plt
# plt.close()
# # di_model.input_current[int(20/dt):] = 0
# # di_model.input_current[:] = 0
# # di_model.plot_input_current(savefig=False)
# di_model.simulate()
# di_model.mass_model.plot(heat_map=True, plot_input=True, plot_combined=True, z_limit=0.12, animate=False, savefig=save_figs)
# import h5py
# with h5py.File(lt_name + '.hdf5', 'r') as h5file:
#     v = np.array(h5file['v'])
#     r_plot = np.array(h5file['r'])
#     for k in range(r_plot.shape[0]):
#         r_plot[k, -1] = r_plot[k, -2]
#     rho_plot = np.array(h5file['rho_plot'])
#     p_types_raw = h5file['p_types']
#     p_types = p_types_raw.asstr()[:]
#     if 'rho' in h5file:
#         rho = np.array(h5file['rho'])
# plt.plot(v, rho[0, :, -1])
# plt.savefig('long_time.png')
# plt.close()
# print('saved long time plot')
# #
# # # extra PA, LM examples
# #
# # parameters['theta'] = 2
# # parameters['name'] = 'CNS_26_example_PA'
# # di_model = DI_wave_simulation(parameters=parameters)
# # di_model.simulate()
# # di_model.labelsize=24
# # di_model.plot_voltage(savefig=save_figs)
# # di_model.mass_model.clean()
# # parameters['theta'] = 130
# # parameters['name'] = 'CNS_26_example_LM'
# # di_model = DI_wave_simulation(parameters=parameters)
# # di_model.simulate()
# # di_model.labelsize=24
# # di_model.plot_voltage(savefig=save_figs)
# # di_model.mass_model.clean()
# #
