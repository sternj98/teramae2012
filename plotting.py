import nest
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import elephant as ele
from elephant.kernels import GaussianKernel
import quantities as pq
import neo

gkernel = GaussianKernel(sigma = 30 * pq.ms)

# not currently working?
def cc(inputSD,outputSD,timerange):
    input_ts = nest.GetStatus(inputSD,keys = "events")[0]["times"]
    output_ts = nest.GetStatus(outputSD,keys = "events")[0]["times"]

    input_rate = len(input_ts) / (timerange[1] - timerange[0])
    output_rate = len(output_ts) / (timerange[1] - timerange[0])

    input_st = neo.SpikeTrain(times = input_ts, units = 'ms',t_start = timerange[0],t_stop = timerange[1])
    output_st = neo.SpikeTrain(times = output_ts, units = 'ms',t_start = timerange[0],t_stop = timerange[1])

    bin_ist = ele.conversion.BinnedSpikeTrain(input_st,t_start = timerange[0]* pq.ms,t_stop = timerange[1]* pq.ms,binsize = 3. * pq.ms)
    bin_ost = ele.conversion.BinnedSpikeTrain(output_st,t_start = timerange[0]* pq.ms,t_stop = timerange[1]* pq.ms,binsize = 3. * pq.ms)

    binst = ele.conversion.BinnedSpikeTrain([input_st,output_st],t_start = timerange[0]* pq.ms,t_stop = timerange[1]* pq.ms,binsize = 3. * pq.ms)
    # ele.spike_train_correlation.covariance(binst)
    # P(out|in)
    p_coincident = len(np.where((bin_ist.to_array() > 0) & (bin_ost.to_array() > 0))[0]) / len(bin_ist.to_array())

    return p_coincident * np.sqrt(input_rate / output_rate)

def heatmapConnectivity(input_pop,output_pop):
    plt.figure()
    connections = nest.GetConnections(input_pop,output_pop)
    weights = nest.GetStatus(connections,["weight"])
    edges = nest.GetStatus(connections,["source","target"])

    C = np.zeros((len(input_pop)+1,len(output_pop)+1))
    C[tuple(np.array(edges).T)] = 1
    sns.heatmap(C)

def plotConnectionDist(input_pop,output_pop):
    plt.figure()
    connections = nest.GetConnections(input_pop,output_pop)
    weights = nest.GetStatus(connections,["weight"])
    mu = np.mean(weights)
    sigma2 = np.var(weights)
    print("Mean weight: %f\n"%mu)
    print("Variance of weights: %f\n"%sigma2)
    bins = np.linspace(0.001,10.0, 100)
    plt.xlim([0.001,10])
    sns.distplot(weights,bins = bins,kde = False)
    plt.xlabel('EPSP (mV)')
    plt.ylabel('Probability density')
    plt.title('Synapse Weight distribution')

def plotConnectionDistLogscale(input_pop,output_pop):
    plt.figure()
    connections = nest.GetConnections(input_pop,output_pop)
    weights = nest.GetStatus(connections,["weight"])
    edges = nest.GetStatus(connections,["source","target"])

    n_inputs = len(weights)
    print("Number of connections:",n_inputs)

    mu = np.mean(weights)
    sigma2 = np.var(weights)
    print("Mean weight: %f\n"%mu)
    print("Variance of weights: %f\n"%sigma2)
    bins = np.logspace(np.log(0.0001),np.log(10.0), 100)
    plt.xscale('log')
    plt.xlim([0.001,10])
    sns.distplot(weights,bins = bins,kde = False)
    plt.xlabel('EPSP (mV)')
    plt.ylabel('Probability density')
    plt.title('Synapse Weight distribution')

def instantaneous_rate(ts,t_stop):
    spiketrain = neo.SpikeTrain(times = ts , units = 'ms', t_stop = t_stop)
    rate = ele.statistics.instantaneous_rate(spiketrain,sampling_period = pq.ms,kernel = gkernel) # sample 1/ms
    return rate.flatten()

def plot_rates(spikedetector,t_stop,colors = [],styles = []):
    plt.figure()
    dSD = nest.GetStatus(spikedetector,keys = "events")
    for i, sd in enumerate(dSD):
        ts = sd["times"]
        rate = instantaneous_rate(ts,t_stop)
        plt.plot(rate,colors[i],linestyle = styles[i],label = "Neuron %i"%i)
    plt.legend()
    plt.ylabel('FR (Hz)')
    plt.xlabel('Time (ms)')
    plt.title('Circuit Firing Rate Activity')

def AvgFRHist(spikedetector,timerange,color = 'r'):
    plt.figure()
    dSD = nest.GetStatus(spikedetector,keys = "events")
    allts = np.array(dSD[0]['times'])
    senders = np.array(dSD[0]['senders'])

    # now filter by timerange
    timerangeMask = (timerange[0] < allts) & (timerange[1] > allts)
    allts = allts[timerangeMask]
    senders = senders[timerangeMask]

    ts = [[] for n in range(np.max(dSD[0]['senders']))]
    for t, source in zip(allts,senders):
        ts[source-1].append(t)
    counts = np.array([len(ts[i]) for i in range(len(ts))])
    if np.min(dSD[0]['senders']) > 0:
        counts = counts[np.min(dSD[0]['senders']):]
    avgFR = 1000.0 * counts / (timerange[1] - timerange[0])
    sns.distplot(avgFR,bins = np.linspace(0,40,40),color = color)
    plt.xlabel('Avg FR (Hz)')
    plt.xlim([0.0,40.0])

def excInhAvgFRHist(excSD,inhSD,timerange):
    plt.figure()
    avgFR = []

    for spikedetector in [excSD,inhSD]:
        dSD = nest.GetStatus(spikedetector,keys = "events")
        allts = np.array(dSD[0]['times'])
        senders = np.array(dSD[0]['senders'])

        # now filter by timerange
        timerangeMask = (timerange[0] < allts) & (timerange[1] > allts)
        allts = allts[timerangeMask]
        senders = senders[timerangeMask]

        ts = [[] for n in range(np.max(dSD[0]['senders']))]
        for t, source in zip(allts,senders):
            ts[source-1].append(t)
        counts = np.array([len(ts[i]) for i in range(len(ts))])
        if np.min(dSD[0]['senders']) > 0:
            counts = counts[np.min(dSD[0]['senders']):]
        avgFR.append(1000.0 * counts / (timerange[1] - timerange[0]))
    sns.distplot(avgFR[0],color = 'r')
    sns.distplot(avgFR[1],color = 'b')
    plt.xlabel('Avg FR (Hz)')

def circuitRasterPlot(spikedetector,timerange = (0.0,0.0),colors = []):
    plt.figure()
    dSD = nest.GetStatus(spikedetector,keys = "events")
    allts = dSD[0]['times']
    senders = dSD[0]['senders']
    ts = [[] for n in range(np.max(dSD[0]['senders']))]
    for t, source in zip(allts,senders):
        ts[source-1].append(t)

    plt.title('Raster plot of circuit activity')
    plt.eventplot(ts,colors = colors)
    plt.xlim(timerange[0],timerange[1])
    plt.xlabel('Time')
    plt.ylabel('Neuron')

def inhExcCircuitRasterPLot(excSD,inhSD,nExc,nInh,timerange = (0.0,0.0)):
    excSD = nest.GetStatus(excSD,keys = "events")
    inhSD = nest.GetStatus(inhSD,keys = "events")
    ets = excSD[0]['times']
    esenders = excSD[0]['senders']
    its = inhSD[0]['times']
    isenders = inhSD[0]['senders']

    allts = list(ets) + list(its)
    senders = list(esenders) + list(isenders)

    ts = [[] for n in range(nExc + nInh)]
    for t, source in zip(allts,senders):
        ts[source-1].append(t)

    ts = list(filter(None, ts)) # take out non spiking neurons
    eUniq = np.unique(esenders)
    iUniq = np.unique(isenders)
    print('Active neurons',len(eUniq) + len(iUniq))

    colors = ['r' for n in range(len(eUniq))] + ['b' for n in range(len(iUniq))]
    plt.figure()
    plt.title('Raster plot of circuit activity')
    plt.eventplot(ts,colors = colors)
    plt.xlim(timerange[0],timerange[1])
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')

def circuitCurrentPlot(multimeter,spikedetector,timerange = (0.0,0.0)):
    plt.figure()
    dmm = nest.GetStatus(multimeter)

    dSD = nest.GetStatus(spikedetector,keys = "events")
    ts = []
    for i, sd in enumerate(dSD):
        ts.append(sd["times"])

    fig, ax = plt.subplots(nrows = len(dmm),ncols = 2)
    for i, mm in enumerate(dmm):
        mm = dmm[i]["events"]
        spikeyV = mm["v"].copy()
        round_ts = np.array([int(round(t)) for t in ts[i]])
        round_ts[np.where(round_ts >= len(spikeyV))[0]] = len(spikeyV) - 1
        if len(round_ts) > 0:
            spikeyV[np.array(round_ts)] = 0 # for visual

        # current plotting
        ax[i,0].plot(mm["times"],mm["gE"],'r',label = "gE")
        ax[i,0].plot(mm["times"],mm["gI"],'b',label = "gI")
        ax[i,0].set_ylabel("Currents")
        # voltage plotting
        ax[i,1].plot(mm["times"],spikeyV,'g',label = "voltage")
        ax[i,1].set_ylabel("Voltage")

    for _ax in ax:
        _ax[0].legend(loc="upper right")
        _ax[0].set_xlim(timerange[0], timerange[1])
        _ax[0].grid(True)
        _ax[0].set_xticklabels([])
        _ax[1].legend(loc="upper right")
        _ax[1].set_xlim(timerange[0], timerange[1])
        _ax[1].grid(True)
        _ax[1].set_xticklabels([])

    ax[-1][0].set_xlabel("time")
    ax[-1][1].set_xlabel("time")
    plt.suptitle("Circuit Current and Voltage Activity")

def singleNeuronPlot(spikedetector,multimeter,timerange):
    dSD = nest.GetStatus(spikedetector,keys="events")[0]
    evs = dSD["senders"]
    ts = dSD["times"]

    dmm = nest.GetStatus(multimeter)[0]
    dmm = dmm["events"] # just need the events part of the multimeter

    fig, ax = plt.subplots(nrows=4)

    print('average v_m: %f', np.mean(dmm["v"]))

    spikeyV = dmm["v"].copy()
    round_ts = np.array([int(round(t)) for t in ts])
    round_ts[np.where(round_ts >= len(spikeyV))[0]] = len(spikeyV) - 1

    if len(round_ts) > 0:
        spikeyV[np.array(round_ts)] = 0

    ax[0].plot(dmm["times"], spikeyV, label="v")
    ax[0].set_ylabel("voltage")

    ax[1].plot(dmm["times"], dmm["gE"], label="gE")
    ax[1].set_ylabel("gE")

    ax[2].plot(dmm["times"], dmm["gI"], label="gI")
    ax[2].set_ylabel("gI")

    ax[3].eventplot(ts,label = "spike times")
    ax[3].set_ylabel("spikes")

    for _ax in ax:
        _ax.legend(loc="upper right")
        _ax.set_xlim(timerange[0], timerange[1])
        _ax.grid(True)

    for _ax in ax[:-1]:
        _ax.set_xticklabels([])

    ax[-1].set_xlabel("time")
