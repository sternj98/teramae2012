## recreation of single neuron input cross correlation figure from teremae et al 2012
# plotting CC for top 3 synapses at varying mean membrane voltages (noise levels)

import nest
import numpy as np
from numpy import random as rnd
from matplotlib import pyplot as plt
import seaborn as sns
from plotting import plotConnectionDistLogscale,singleNeuronPlot,cc

nest.set_verbosity("M_WARNING")
nest.Install("nestmlmodule")
nest.ResetKernel()
# msd = 1000
msd = np.random.randint(1000)
N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
pyrngs = [np.random.RandomState(s) for s in range(msd, msd+N_vp)]
nest.SetKernelStatus({"grng_seed" : msd+N_vp})
nest.SetKernelStatus({"rng_seeds" : range(msd+N_vp+1, msd+2*N_vp+1)})

# set up neuron to record from
output_neuron = nest.Create("teremaeNeuron") # automatically exc

# add recording devices
output_mm = nest.Create("multimeter")
nest.SetStatus(output_mm, {"withtime": True , "record_from": ["v","gE","gI"]})
output_sd = nest.Create("spike_detector",params={"withgid": True, "withtime": True})

nest.Connect(output_mm, output_neuron)
nest.Connect(output_neuron, output_sd)

# set up lognormal inputs
# limitation of this setup: inputs independent from one another--> runaway excitation
# the solution is to pursue the full simulation
n_inputs = 500
rate = 1.5
input_pop = nest.Create("poisson_generator",n_inputs)
nest.SetStatus(input_pop,[{"rate": rate, "start": 100.0, "stop": 100000.0} for n in range(n_inputs)])

inh_pop = nest.Create("poisson_generator",n_inputs)
nest.SetStatus(inh_pop,[{"rate": rate * 2.0, "start": 100.0, "stop": 100000.0} for n in range(n_inputs)])


# now connect excitatory and inhibitory batteries
wMax = 20.0
sigma = 1.0
mu = (1.0 + np.log(0.2))
wEI = 1.5 # 1.5 works #.0180 / .0098 # 1.5 # 0.0180 ... maybe determine real epsp dist w/ numpy dist instead of this shit
# wEI = 2.0
wIE = -.0018 # -.0020 # -0.0020
# wIE = -2.0
wII = -0.0025 # 0
pI = 0.5

lognormalDist = rnd.lognormal(mu,sigma,(1,n_inputs))
lognormalDist[(lognormalDist > wMax)] = wMax

largest3 = np.argpartition(lognormalDist.flatten(),-3)[-3:]

# # alter weak synapse strength
# print(lognormalDist)
# print((lognormalDist < 1.0) & (lognormalDist < 3.0))
# lognormalDist[(lognormalDist > .5) & (lognormalDist < 3.0)] = 1.0
# print(lognormalDist)

nest.CopyModel('static_synapse','excitatory')
ee_syn_dict = {"model": "excitatory",
               "weight":lognormalDist,
               "delay": {"distribution": "uniform", "low": 1.0, "high": 3.0}}

# ee_syn_dict = {"model": "excitatory",
#                "weight": {"distribution": "lognormal_clipped", "low": 0.01,"high": wMax, "mu": mu, "sigma": sigma},
#                "delay": {"distribution": "uniform", "low": 1.0, "high": 3.0}}

nest.CopyModel('static_synapse','inhibitory')
ie_syn_dict = {"model": "inhibitory",
               "weight": wIE,
               "delay": {"distribution": "uniform", "low": 0.1, "high": 2.0}}

parrot_inputs = nest.Create('parrot_neuron',n_inputs)
nest.Connect(input_pop,parrot_inputs,'one_to_one', {'model': 'static_synapse','weight': 1.0})

nest.Connect(parrot_inputs,output_neuron,syn_spec = ee_syn_dict)
nest.Connect(inh_pop,output_neuron,syn_spec = ie_syn_dict)

# measure spikes from the strongest inputs
inputSD = []
for id in largest3:
    parrot = nest.Create('parrot_neuron')
    nest.Connect([parrot_inputs[id]],parrot)
    sd = nest.Create("spike_detector",params={"withgid": True, "withtime": True})
    nest.Connect(parrot,sd)
    inputSD.append(sd)

plotConnectionDistLogscale(parrot_inputs,output_neuron) # check distribution

nest.Simulate(1000.0)

print(largest3)
for sd in inputSD:
    print(cc(sd,output_sd,(0.,1000.)))

singleNeuronPlot(output_sd,output_mm,(0.0,1000.0))
plt.show()
