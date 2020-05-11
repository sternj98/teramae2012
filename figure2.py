import numpy as np
import random
import nest
import nest.raster_plot
from matplotlib import pyplot as plt
from plotting import singleNeuronPlot,plotConnectionDistLogscale,heatmapConnectivity,inhExcCircuitRasterPLot,AvgFRHist,excInhAvgFRHist

# a reproduction of figure 2 of Teremae, Tsubo, Fukai 2012
nest.Install("nestmlmodule")
nest.ResetKernel()

# msd = 1000
msd = np.random.randint(1000)
N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
pyrngs = [np.random.RandomState(s) for s in range(msd, msd+N_vp)]
nest.SetKernelStatus({"grng_seed" : msd+N_vp})
nest.SetKernelStatus({"rng_seeds" : range(msd+N_vp+1, msd+2*N_vp+1)})

# create excitatory and inhibitory batteries
tau_exc = 20.0 # membrane time constant for exc neuron
tau_inh = 10.0 # membrane time constant for inhibitory neuron

nExc = 10000
nInh = 2000

exc = nest.Create("teremaeNeuron",nExc,[{"tau_m":tau_exc}])
inh = nest.Create("teremaeNeuron",nInh,[{"tau_m":tau_inh}])

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
pE = 0.01

# inhibition not working right lol
nest.CopyModel('bernoulli_synapse','ee')
nest.CopyModel('static_synapse','ei')
nest.CopyModel('static_synapse','ie')
nest.CopyModel('static_synapse','ii')

ee_syn_dict = {"model": "ee","p_transmit":0.95, # 0.6
               "weight": {"distribution": "lognormal_clipped", "high": wMax, "mu": mu, "sigma": sigma},
               "delay": {"distribution": "uniform", "low": 1.0, "high": 3.0}}
ee_conn_dict = {'rule': 'pairwise_bernoulli', 'p': pE,'autapses':False}

ei_syn_dict = {"model": "ei",
               "weight": wEI,
               "delay": {"distribution": "uniform", "low": 0.1, "high": 2.0}}
ei_conn_dict = {'rule': 'pairwise_bernoulli', 'p': pE,'autapses':False}

ie_syn_dict = {"model": "ie",
               "weight": wIE,
               "delay": {"distribution": "uniform", "low": 0.1, "high": 2.0}}
ie_conn_dict = {'rule': 'pairwise_bernoulli', 'p': pI,'autapses':False}

ii_syn_dict = {"model": "ii",
               "weight": wII,
               "delay": {"distribution": "uniform", "low": 0.1, "high": 2.0}}
ii_conn_dict = {'rule': 'pairwise_bernoulli', 'p': pI,'autapses':False}

nest.Connect(exc,exc,ee_conn_dict,ee_syn_dict)
nest.Connect(exc,inh,ei_conn_dict,ei_syn_dict)
nest.Connect(inh,exc,ie_conn_dict,ie_syn_dict)
nest.Connect(inh,inh,ii_conn_dict,ii_syn_dict)

# create poisson noise
noiseExc = nest.Create('poisson_generator',nExc)
noiseInh = nest.Create('poisson_generator',nInh)
# don't know rate, should be "to every neuron" ie one-to-one
nest.SetStatus(noiseExc,[{"rate": 90.0, "origin": 0.0, "start": 0.0, "stop": 100.0}])
nest.SetStatus(noiseInh,[{"rate": 90.0, "origin": 0.0, "start": 0.0, "stop": 100.0}])
nest.Connect(noiseExc,exc,"one_to_one",{"weight": 7.})
nest.Connect(noiseInh,inh,"one_to_one",{"weight": 8.})
# nest.SetStatus(noiseExc,[{"rate": 10.0, "origin": 0.0, "start": 0.0, "stop": 100.0}])
# nest.SetStatus(noiseInh,[{"rate": 10.0, "origin": 0.0, "start": 0.0, "stop": 100.0}])
# nest.Connect(noiseExc,exc,"one_to_one",{"weight": 10.})
# nest.Connect(noiseInh,inh,"one_to_one",{"weight": 10.})

sample = random.sample(list(range(nExc)),k = 5)
mms = []
sds = []
for id in sample:
    mm = nest.Create("multimeter")
    mm = nest.Create("multimeter")
    nest.SetStatus(mm, {"withtime": True , "record_from": ["v","gE","gI"]})
    sd = nest.Create("spike_detector",params={"withgid": True, "withtime": True})

    nest.Connect(mm,[exc[id]])
    nest.Connect([exc[id]],sd)
    sds.append(sd)
    mms.append(mm)

# set devices for recording
excSD = nest.Create("spike_detector")
nest.Connect(exc, excSD)
inhSD = nest.Create("spike_detector")
nest.Connect(inh, inhSD)

# plot connection distribution
# plotConnectionDistLogscale(exc,exc)
# heatmapConnectivity(exc,exc)
# plt.show()

# perform the simulation
nest.Simulate(1000.0)

excInhAvgFRHist(excSD,inhSD,(100.0,1000.0))
# excInhAvgFRHist(excSD,inhSD(100.0,1000.0))
# AvgFRHist(inhSD,(100.0,1000.0),color = 'b')

# plt.show()
# single neuron data
for i in range(5):
    singleNeuronPlot(sds[i],mms[i],[0.0,1000.])
# plt.show()

inhExcCircuitRasterPLot(excSD,inhSD,nExc,nInh,[0.0,1000.0])
plt.title('Population dynamics')
plt.show()
