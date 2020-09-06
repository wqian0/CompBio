import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as sp
import pickle
from scipy import io
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D


mat = sp.io.loadmat('C:/Users/billy/PycharmProjects/PHYS280updated/contrast_response.mat')

contrasts = np.array([x[0] for x in mat["contrasts"]])
spikeTimes = mat["spikeTimes"][0]
spikeTemp = []
for i in range(len(spikeTimes)):
    spikeTemp.append(np.array([x[0] for x in spikeTimes[i]]))
spikeTimes = spikeTemp
stimulus = np.array([x[0] for x in mat["stimulus"]])

def gaus(x, x0,sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))

def get_firing_rate(spikes):
    hist = plt.figure(2)
    A, B, C = plt.hist(spikes, bins=1000, range=(0, 200000))
    plt.close()
    return A/.1

def get_smooth_firing_rate(spikes, sigma):
    input_times = np.linspace(0, 200000, 200000)
    output = np.zeros(200000)
    for i in range(len(spikes)):
        gaussian = gaus(input_times, spikes[i], sigma)
        gaussian = 2000 * gaussian / np.sum(gaussian)
        output += gaussian
    return output

def create_STA_plot(spikes, stimulus, tau, fig_num, contrast, show_plot = True):
    output = np.zeros(tau)
    padding = np.zeros(tau)
    padded_stimulus = np.concatenate((padding, stimulus * contrast))
    for i in range(len(spikes)):
        ## plus instead of minus here due to padding shifting indices
        output += padded_stimulus[spikes[i]:spikes[i] + tau]
    output /= len(spikes)
    output /= np.sqrt(np.sum(output * output))
    if show_plot:
        plot = plt.figure(fig_num)
        plt.title("Spike-Triggered Average for contrast = " + str(contrast))
        plt.xlabel("time (s)")
        plt.ylabel("Spike Triggered Average")
        plt.plot(np.linspace(0, tau / 2000, tau), output, label = '%.3f'%contrast)
        plt.tight_layout()
    return output

def create_trigger_intensity_plot(stimulus, STA, plot_time, fig_num, contrast, padding, scaling = False, remove_negs = False, show_plot = True):
    output = np.convolve(contrast * stimulus, list(reversed(STA)), mode = 'same')
    if remove_negs:
        output[output < 0] = 0
    output = np.concatenate((np.zeros(padding), output))
    if show_plot:
        plt.figure(fig_num)
        plt.title("Trigger-feature Intensity for contrast = "+str(contrast))
        plt.xlabel("time (s)")
        plt.ylabel("Trigger feature intensity")
        if scaling:
            plt.plot(np.linspace(0, plot_time/ 2000, plot_time), output[0:plot_time]/np.max(output[0:plot_time]), label = "trigger intensity")
        else:
            plt.plot(np.linspace(0, plot_time / 2000, plot_time), output[0:plot_time])
        plt.tight_layout()
    return output

def plot_2c(trigger_intensities, firing_rates, fig_num, contrast):
    fig = plt.figure(fig_num)
    plt.title("Filter output vs firing rate for contrast = "+str(contrast))
    plt.xlabel("Filter output")
    plt.ylabel("Firing rate (spikes/s)")
    plt.scatter((trigger_intensities[0:200000])[0::10], firing_rates[0::10], s = 10)
    plt.tight_layout()
    plt.savefig(str(contrast)+".png")

def plot_2d(trigger_intensities, firing_rates, x_vals, fig_num, contrast, show_plot = True):
    trigger_intensities = (trigger_intensities[0:200000])[0::10]
    firing_rates = firing_rates[0::10]
    points = []
    x_comps = np.zeros(x_vals)
    y_comps = np.zeros(x_vals)
    counts = np.zeros(x_vals)
    min = np.min(trigger_intensities)
    max  = np.max(trigger_intensities)
    bin_size = (max - min +.01) / x_vals
    ## .01 is to address the last element falling out of bounds
    for i in range(len(trigger_intensities)):
        index = int((trigger_intensities[i] - min)/ bin_size)
        x_comps[index] += trigger_intensities[i]
        y_comps[index] += firing_rates[i]
        counts[index] += 1

    x_comps /= counts
    y_comps /= counts
    if show_plot:
        fig = plt.figure(fig_num)
        plt.title("Averaged trigger intensity vs firing rate (contrast = " + str(contrast) +")")
        plt.xlabel("Filter output")
        plt.ylabel("Firing rate (spikes/s)")
        plt.scatter(x_comps, y_comps, s = 10)
        plt.tight_layout()
        plt.savefig(str(contrast) + " 2d.png")
    return x_comps, y_comps

def plot_2e(trigger_intensities, firing_rates, smooth_interp, fig_num, contrast):
    output = smooth_interp(trigger_intensities[0:40000])
    fig = plt.figure(fig_num)
    plt.title("Actual vs Predicted firing rate (contrast = "+str(contrast)+")")
    plt.xlabel("time (s)")
    plt.ylabel("Firing rate (spikes/s)")
    t = np.linspace(0, 20, 40000)
    plt.plot(t, output, label = 'predicted firing rate', linewidth = .3)
    plt.plot(t, firing_rates[0:40000], label = 'actual firing rates', linewidth = .3)
    plt.legend()
    plt.savefig(str(contrast) + " 2e.png")


##1a)

plot_1a = plt.figure(1)
plt.title("Highest contrast trial timeseries")
plt.xlabel("time (s)")
plt.ylabel("response")
high_contrast = spikeTimes[8]

t = np.linspace(0, 20, 200000)
t_plot = np.linspace(0, 20, 40000)
spikes_to_plot = []

for i in range(len(high_contrast)):
    if high_contrast[i] < 40000:
        spikes_to_plot.append(high_contrast[i])
    else:
        break

binary_spikes = np.zeros(40000)
for i in range(len(spikes_to_plot)):
    binary_spikes[spikes_to_plot[i]] = 1
plt.plot(t_plot, binary_spikes, linewidth = .1)

##1b)

firing_rate_1b = plt.figure(3)
plt.title("Time-dependent firing rate")
plt.xlabel("time (s)")
plt.ylabel("firing rate (spikes/s)")
plt.plot(np.linspace(0, 20, 200), get_firing_rate(spikeTimes[8])[0:200])

## 1c)

firing_rate_1c = plt.figure(4)
plt.title("Smooth Time-dependent Firing Rate")
plt.xlabel("time (s)")
plt.ylabel("firing rate (spikes/s)")
plt.plot(t_plot, get_smooth_firing_rate(spikeTimes[8], 70)[0:40000])


## 1d)
output = np.zeros(len(spikeTimes))
for i in range(len(spikeTimes)):
    output[i] = np.mean(get_firing_rate(spikeTimes[i]))

plt.figure(5)
plt.title("Overall average firing rate as a function of contrast")
plt.xlabel("Contrast")
plt.ylabel("Overall firing rate (spikes/s)")
plt.plot(contrasts, output)

## 1e)

STAs = []
firing_rates = []
trigger_intensities = []
for i in range(len(spikeTimes)):
    STAs.append(create_STA_plot(spikeTimes[i], stimulus, 800, 6+i, contrasts[i], show_plot= False))
    #plt.legend()
    firing_rates.append(get_smooth_firing_rate(spikeTimes[i], 55))


for i in range(len(spikeTimes)):
    trigger_intensities.append(create_trigger_intensity_plot(stimulus, STAs[i], 40000, 15+i, contrasts[i], 400,scaling = False, remove_negs = False, show_plot= True))

## 2b)
create_trigger_intensity_plot(stimulus, STAs[8], 40000, 24, contrasts[8], 400, scaling = True, remove_negs = True, show_plot= True)
plt.plot(np.linspace(0, 20, 40000), firing_rates[8][0:40000]/np.max(firing_rates[8][0:40000]), label = "actual firing rate")
plt.legend()


x_coords = []
y_coords = []
smooth_interps = []

for i in range(len(spikeTimes)):
    plot_2c(trigger_intensities[i], firing_rates[i], 25 + i, contrasts[i])

for i in range(len(spikeTimes)):
    x_comp, y_comp = plot_2d(trigger_intensities[i], firing_rates[i], 30, 34 + i, contrasts[i], show_plot= False)
    x_coords.append(x_comp)
    y_coords.append(y_comp)
    smooth_interps.append(interp1d(x_comp, y_comp, bounds_error= False, fill_value="extrapolate"))

for i in range(len(spikeTimes)):
    plot_2e(trigger_intensities[i], firing_rates[i], smooth_interps[i], 43 + i, contrasts[i])


plt.show()