import nest
from plotting import singleNeuronPlot,circuitRasterPlot,circuitCurrentPlot,plot_rates
from matplotlib import pyplot as plt
import numpy as np

# play around with a few neurons to test communication properties
# maybe make a simple FFI circuit

nest.set_verbosity("M_WARNING")
nest.Install("nestmlmodule")
nest.ResetKernel()
# msd = 1000
msd = np.random.randint(1000)
N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
pyrngs = [np.random.RandomState(s) for s in range(msd, msd+N_vp)]
nest.SetKernelStatus({"grng_seed" : msd+N_vp})
nest.SetKernelStatus({"rng_seeds" : range(msd+N_vp+1, msd+2*N_vp+1)})

tau_exc = 20.0 # membrane time constant for exc neuron
tau_inh = 10.0 # membrane time constant for inhibitory neuron
tau_list = [tau_exc, tau_inh, tau_exc]

# excitatory -> inhibitory -| excitatory
spec_list = [{"tau_m":tau} for tau in tau_list]
circuit = nest.Create("teremaeNeuron",3,spec_list)

# add recording devices
multimeter = nest.Create("multimeter",3)
nest.SetStatus(multimeter, {"withtime": True , "record_from": ["v","gE","gI"]})
spikedetector = nest.Create("spike_detector",3,params={"withgid": True, "withtime": True})

nest.Connect(multimeter, circuit,"one_to_one")
nest.Connect(circuit, spikedetector,"one_to_one")

# add baseline poisson input to neuron 1 and 3
# base_input = nest.Create('poisson_generator',3)
baseline_input = nest.Create('poisson_generator')
secondary_input = nest.Create('poisson_generator')
nest.SetStatus(baseline_input,[{"rate": 50.0, "origin": 0.0, "start": 0.0, "stop": 900.0}])
nest.SetStatus(secondary_input,[{"rate": 50.0, "origin": 0.0, "start": 300.0, "stop": 900.0}])
nest.CopyModel("static_synapse","excitatory",{"weight":5., "delay":0.5})
# nest.Connect(baseline_input,circuit,"one_to_one",syn_spec = "excitatory")
nest.Connect(baseline_input,[circuit[2]],syn_spec = "excitatory")
# !! going to need to alter teremaeNeuron model to get multiple types of exc/inh input
nest.Connect(secondary_input,[circuit[0]],syn_spec = "excitatory")

# connect neuron1 and neuron2
nest.Connect([circuit[0]],[circuit[1]],syn_spec = "excitatory")

# connect neuron2 and neuron3 to create FFI
nest.CopyModel("static_synapse","inhibitory",{"weight":-5.,"delay":0.5})
nest.Connect([circuit[1]],[circuit[2]],syn_spec = "inhibitory")

nest.Simulate(1000.0)

# plot circuit activity
colors = ['r','b','r'] # determined by neuron type
styles = ['-','-',':']
circuitRasterPlot(spikedetector,timerange = (0.0,1000.0),colors = colors)
circuitCurrentPlot(multimeter,spikedetector,timerange = (0.0,1000.0))
plot_rates(spikedetector,1000.0,colors,styles = styles)
plt.show()
