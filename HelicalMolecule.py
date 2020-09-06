import numpy as np
import matplotlib.pyplot as plt
import cmath

bp_per_pitch = 10.5
rise_per_bp = .34
lamba = .05
def v_function(u):
    return np.sin(2*np.pi *u/(bp_per_pitch*rise_per_bp))

def p_amplitude(x_bar, y_bar, lamba, u_locs, v_locs):
    const_x = -1j * 2*np.pi*x_bar/lamba
    const_y = -1j * 2*np.pi*y_bar/lamba
    vec_result = np.exp(const_x * u_locs + const_y * v_locs )
    return np.abs(np.sum(np.real(vec_result))+1j*np.sum(np.imag(vec_result))) ** 2


u_locs = np.linspace(-24.82, 24.82, 147)
print(u_locs)
v_locs = v_function(u_locs)

scattering_pts = plt.figure(1)
plt.title("Scattering Points")
plt.xlabel("u (nm)")
plt.ylabel("v (nm)")
plt.plot(u_locs, v_locs, linewidth = 1)
plt.scatter(u_locs, v_locs, color = 'black', s=10)

output_vals = np.zeros((201, 201))
for r in range(len(output_vals)):
    for c in range(len(output_vals)):
        output_vals[r][c] = p_amplitude(.15 - r * .0015, -.15 + c * .0015, lamba, u_locs, v_locs)


heat_map = plt.figure(2)
plt.title("Intensity Map of PDF as a Function of x/d, y/d")
plt.xlabel("y/d")
plt.ylabel("x/d")
plt.xticks(np.linspace(-.15, .15, 6), fontsize = 9)
plt.yticks(np.linspace(-.15, .15, 6), fontsize = 9)
plt.imshow(output_vals, cmap = 'Greys_r', extent=[-.15, .15,-.15,.15], interpolation= 'nearest', vmax = 5000)
plt.colorbar()


plt.show()

