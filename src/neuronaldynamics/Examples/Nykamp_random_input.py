import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from Model.Nykamp_Model import Nykamp_Model_1

matplotlib.use('TkAgg')

dt = 0.01
T = 100
t = np.arange(0, T, dt)
dv = 0.1
random_signal_exc = np.random.rand(t.shape[0])
random_signal_exc = 7*(random_signal_exc)

random_signal_inh = -np.random.rand(t.shape[0])
random_signal_inh = 1*(random_signal_inh)
#TODO: exc in to exc input idx, inh o inh ...


pars = {'dt': dt, 'T': T, 'input_function': random_signal_exc, 'multiple_inputs':True, 'input_function_idx':[[0, 0],
                                                                                                             [0, 1]]}
nyk1D = Nykamp_Model_1(parameters=pars, name='nyk_random')

nyk1D.simulate()
ps = nyk1D.p[0]
vs = nyk1D.v
ps_end = ps[:, -1000:]
std_end = np.std(ps_end, axis=0)
mean_end = np.mean(ps_end)
mean_std = np.mean(std_end)
print(f'mean std {mean_std}')
print(f'mean {mean_end}')
for i in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 980]:
    plt.plot(vs, ps_end[:, i])
plt.show()
nyk1D.plot(heat_map=True, plot_input=False)
nyk1D.save_log()
nyk1D.clean()





