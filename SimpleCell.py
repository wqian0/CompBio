import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as sp
import pickle
from scipy import io
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D

def gabor_spatial(x, y, sig_x, sig_y, k, phi):
    return 1/(2 * np.pi * sig_x * sig_y) * np.exp(-x ** 2 / (2*sig_x ** 2) - y ** 2/ (2 * sig_y ** 2)) * np.cos(k * x - phi)
def gabor_temp(time, alpha):
    return alpha * np.exp(-alpha * time)*((alpha * time) ** 5 / 120  - (alpha * time) ** 7 / 5040)

def grating(x, y, K, Theta, Phi, omega, t):
    return np.cos(K*x*np.cos(Theta)+K*y*np.sin(Theta) - Phi)*np.cos(omega * t)

def get_spatial_response(x, y, dx, dy, sig_x, sig_y, k, phi, K, Theta, Phi):
    return dx*dy*np.sum(gabor_spatial(x, y, sig_x, sig_y, k, phi) * grating(x, y, K, Theta, Phi, 0, 0))

def get_temporal_response(time, tau, dtau, alpha, omega):
    return dtau * np.sum(gabor_temp(tau, alpha)*np.cos(omega*(time - tau)))


v = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(v, v)
spatial = gabor_spatial(X, Y, 2, 2, 1, 0)
print(spatial.shape)

##3a)
fig = plt.figure(1)
ax = Axes3D(fig)
ax.set_title("Spatial receptive field")
ax.set_xlabel("x (degrees)")
ax.set_ylabel("y (degrees)")
ax.set_zlabel("D_s")

p = ax.plot_surface(X, Y, spatial, rstride = 1, cstride = 1, cmap = "jet")
fig.colorbar(p)

##3b)

time = np.linspace(0, 300, 1000)
temporal = gabor_temp(time, 1/15) ## units of 1/ms
fig_temp = plt.figure(2)
plt.title("Temporal structure of receptive field")
plt.xlabel("tau (ms)")
plt.ylabel("D_t (Hz)")
plt.plot(time, temporal * 1000) # *1000 to convert to Hz


##3c)

for i in range(6):
    tau = 40 + 40 * i
    gabor = gabor_temp(tau, 1/15) * gabor_spatial(X, Y, 2,2,1,0)
    plt.figure(3+i)
    plt.xlabel("x (degrees)")
    plt.ylabel("y (degrees)")
    plt.title("Spatiotemporal dynamics (Hz / deg ^2 )at tau = "+str(tau))
    plt.contourf(v,v, gabor * 1000, levels = 60)
    plt.colorbar()

##3d)
K = 10
Theta = np.pi/12
Phi = 0
omega = 10
time = np.linspace(0, 300, 1000)
heat_map = plt.figure(9)
plt.xlabel("x (degrees)")
plt.ylabel("y (degrees)")
plt.title("Grating")

plt.xticks(np.linspace(-5, 5, 11), fontsize = 9)
plt.yticks(np.linspace(-5, 5, 11), fontsize = 9)

output_vals = np.flipud(grating(X, Y, K, Theta, Phi, omega, 0)) # image is flipped to account for array indexing order
print(output_vals.shape)
plt.imshow(output_vals,extent=[-5, 5,-5, 5], vmax= .9, vmin= -.9)
plt.colorbar()

##movie

for i in range(len(time)):
    plt.imshow(np.flipud(grating(X, Y, K, Theta, Phi, omega, time[i])), extent=[-5, 5,-5, 5], vmax= .9, vmin= -.9)
    plt.pause(.1)


##3e)
Theta = np.linspace(-1.5, 1.5, 100)
spatial_response = np.zeros(len(Theta))
dx = v[1]-v[0]
for i in range(len(Theta)):
    spatial_response[i] = get_spatial_response(X, Y, dx, dx, 2, 2, 1, 0, 1, Theta[i], 0)
varying_theta = plt.figure(10)
plt.title("L_s as a function of Theta")
plt.xlabel("Theta")
plt.ylabel("L_s")
plt.plot(Theta, spatial_response)

varying_K = plt.figure(11)
plt.title("L_s as a function of K/k")
plt.xlabel("K/k")
plt.ylabel("L_s")
K_vals = np.linspace(0,3, 100)
spatial_response = np.zeros(len(K_vals))
for i in range(len(Theta)):
    spatial_response[i] = get_spatial_response(X, Y, dx, dx, 2, 2, 1, 0, K_vals[i],0, 0)
plt.plot(K_vals, spatial_response)

varying_Phi = plt.figure(12)
plt.title("L_s as a function of Phi")
plt.xlabel("Phi")
plt.ylabel("L_s")
Phi_vals = np.linspace(-3, 3, 100)
spatial_response = np.zeros(len(Phi_vals))
for i in range(len(Phi_vals)):
    spatial_response[i] = get_spatial_response(X, Y, dx, dx, 2, 2, 1, 0, 1, 0, Phi_vals[i])
plt.plot(Phi_vals, spatial_response)


##3f)
omega = np.linspace(0, 2*np.pi, 50) * .02
output = np.zeros(len(omega))
time = np.linspace(0, 3000, 10000)
tau = np.linspace(0, 1000, 10000)

for i in range(len(omega)):
    print(i)
    response = np.zeros(len(time))
    for j in range(len(time)):
        response[j] = get_temporal_response(time[j], tau, .1, 1/15, omega[i])
    output[i] = np.max(response)

plt.figure(13)
plt.title("Frequency Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.plot(np.linspace(0, 20, len(output)), output)

plt.show()
