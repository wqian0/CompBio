import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import io
from scipy.optimize import curve_fit
V_threshold = 0
spiketime_threshold = 10

mat = sp.io.loadmat('C:/Users/billy/PycharmProjects/Python280/orientation_tuning_data.mat')
print(mat)

Vm = mat["Vm"][0]
stimuli = mat["Stimuli"]


time = np.linspace(1, 10000, 10000)

init_second = plt.figure(1) # plot of first second
plt.plot(time, Vm[0:10000], linewidth = .5)
plt.title("Voltage over time")
plt.xlabel("time (10 ^-4 s)")
plt.ylabel("Voltage (mV)")


spike_times = []
prev_spike_time = 1 - spiketime_threshold
#tracking previous spike time to determine whether successive spikes meet timegap threshold

#5739 accounts for the time of first presentation of stimulus
for i in range(5739, len(Vm)):
    if Vm[i] > V_threshold and i - prev_spike_time > spiketime_threshold:
        spike_times.append(i)
        prev_spike_time = i
spike_times = np.array(spike_times)

#Computing total presentation time and spike counts for each stimuli presentation
spike_counts = np.zeros(16)
total_time = np.zeros(16)
for i in range(len(stimuli)):
    angle_index = stimuli[i][0]
    start = stimuli[i][1]
    if i != len(stimuli) - 1:
        end = stimuli[i+1][1]
    else: # accounts for the very last stimuli presentation.
        end = len(Vm)
    if angle_index != 16: #ignoring the "spontaneous activity" stimuli.
        total_time[angle_index] += end - start
        spike_counts[angle_index] += np.count_nonzero((start < spike_times) & (spike_times < end))

#Determining overall firing rates
firing_rates = np.zeros(16)
for i in range(len(firing_rates)):
    firing_rates[i] = spike_counts[i] / total_time[i]

firing_rates *= 10 ** 4 #scaling from kHz to Hz
angles = np.arange(16) * 22.5
firing_v_stimuli = plt.figure()
plt.plot(angles, firing_rates)
plt.title("Firing rate vs Angle of Stimulus")
plt.xlabel("Angle (degrees)")
plt.ylabel("Firing rate (Hz)")

#gaussian fit function
def gaus(x,a,x0,sigma, offset):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + offset

#restricting to the gaussian-fittable range
n = np.sum(firing_rates[4:13])
mean = np.sum(angles[4:13] * firing_rates[4:13])/n
std = np.sqrt(np.sum(firing_rates[4:13]*(angles[4:13] - mean)**2)/n)

popt,pcov = curve_fit(gaus,angles[4:13],firing_rates[4:13],p0=[angles[8], mean, std, 0.0])

angles = np.linspace(4,13,100)*22.5
plt.plot(angles,gaus(angles,*popt), label='fit')

print(popt)


plt.show()