import nest
import numpy as np
import matplotlib.pyplot as plt
import nest.raster_plot
from pynestml.frontend.pynestml_frontend import to_nest, install_nest
from plotting import singleNeuronPlot

# to_nest(input_path="/Users/joshstern/atomProjects/UchidaLab/nest/nestml/models/teremaeNeuron.nestml", target_path="/Users/joshstern/atomProjects/UchidaLab/nest/nestml/target", logging_level="INFO")
# # # to_nest(input_path="/Users/joshstern/atomProjects/UchidaLab/nest/nestml/models/rc_neuron.nestml", target_path="/Users/joshstern/atomProjects/UchidaLab/nest/nestml/target", logging_level="INFO")
# install_nest("/Users/joshstern/atomProjects/UchidaLab/nest/nestml/target", "/anaconda3/envs/NESTENV")

nest.set_verbosity("M_WARNING")
nest.Install("nestmlmodule")
nest.ResetKernel()

# msd = 1000
msd = np.random.randint(1000)
N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
pyrngs = [np.random.RandomState(s) for s in range(msd, msd+N_vp)]
nest.SetKernelStatus({"grng_seed" : msd+N_vp})
nest.SetKernelStatus({"rng_seeds" : range(msd+N_vp+1, msd+2*N_vp+1)})

neuron = nest.Create("teremaeNeuron")

multimeter = nest.Create("multimeter")
nest.SetStatus(multimeter, {"withtime": True , "record_from": ["v","gE","gI"]})
spikedetector = nest.Create("spike_detector",params={"withgid": True, "withtime": True})

nest.Connect(multimeter, neuron)
nest.Connect(neuron, spikedetector)

# input = nest.Create('poisson_generator')
# nest.SetStatus(input,[{"rate": 20.0, "origin": 0.0, "start": 50.0, "stop": 400.0}])

input = nest.Create('spike_generator',params = {"spike_times":[20.,80.,82.,100.]})

nest.CopyModel("static_synapse","exc",{"weight": .89})
nest.Connect(input,neuron,syn_spec = "exc") # weight is v incr for delta

input_inh = nest.Create('spike_generator',params = {"spike_times":[40.,60.,120.]})
# nest.SetStatus(input_inh,[{"rate": 20.0, "origin": 0.0, "start": 200.0, "stop": 400.0}])
# nest.CopyModel("static_synapse","inhibitory",{"weight":.2, "delay":0.1})
nest.CopyModel("static_synapse", "inh", {"weight": -.5})
# syn_dict = {'model' : 'inhibitory', 'weight':.2, 'delay': 0.1}
nest.Connect(input_inh,neuron,syn_spec = "inh") # weight is v incr for delta

nest.Simulate(200.)

singleNeuronPlot(spikedetector,multimeter,[0.0,200.])

plt.show()
