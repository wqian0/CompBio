import numpy as np
import matplotlib.pyplot as plt
import cmath
import math

L_0 = 10
n_w = 1.34
d = 0.024
R_c = 7.8 * 10 ** -3
lamba = 500 * 10 **(-9)

def q(L_0, R_c, u):
    return L_0 + R_c - np.sqrt(R_c **2 - u **2)

def L_1(L_0, R_c, u):
    return np.sqrt(q(L_0, R_c, u) **2 + u **2)

def L_2(u, x, d, R_c):
    delta = R_c - np.sqrt(R_c **2 - u**2)
    return np.sqrt((u-x) **2 + (d - delta) **2)

def phase_func(L_0, R_c, u, x, d, n_w, lamba):
    return 2*np.pi*(L_1(L_0, R_c, u) + n_w * L_2(u, x, d, R_c))/lamba

def focal_length(R_c, n_w):
    return R_c/(n_w-1)

def focus_dist(f, L_0, n_w):
    return n_w/((1/f)- 1/L_0)

u_vals = np.linspace(-3, 3, 100) * 10 ** -3

x_vals = np.linspace(0, 3, 5) * 10 ** -4

print(focus_dist(focal_length(R_c, n_w), L_0, n_w))
print(L_0)
print(L_1(L_0, R_c, u_vals))
print(L_2(0, 0, d, R_c))
fig_unfocused = plt.figure(1)
plt.title("Phase function for unfocused eye")
plt.xlabel("u (m)")
plt.ylabel("phase function")
for i in range(len(x_vals)):
    result = phase_func(L_0, R_c, u_vals, x_vals[i], d, n_w, lamba)
    x_val_current = np.round( x_vals[i] * 100000)/100000
    plt.plot(u_vals, result, label = "x = " + str(x_val_current))
    plt.axvline(x = -.00025, color = "red")
    plt.axvline(x = .00025, color = "red")
plt.legend()

fig_focused = plt.figure(2)
plt.title("Phase function for focused eye")
plt.xlabel("u (m)")
plt.ylabel("phase function")
d = 0.030811
for i in range(len(x_vals)):
    result = phase_func(L_0, R_c, u_vals, x_vals[i], d, n_w, lamba)
    x_val_current = np.round( x_vals[i] * 10 ** 9)/(10 ** 9)
    plt.plot(u_vals, result, label = "x = " + str(x_val_current))
plt.legend()


plt.show()