import numpy as np
import matplotlib.pyplot as plt

#constants
g_L, g_K, g_Na = .003, .36 , 1.2
E_L, E_K, E_Na, E_rest = -54.387, -77, 50, -65 # mV
c_m = 10 * 10 ** -3 # microF/mm^2
A = .1 #mm^2
dt = .001 # ms
t_final = 15 #ms


## Defining all given functions
def alpha_n(V):
    return .01 * (V + 55)/(1 - np.exp(-.1 * (V + 55)))
def alpha_m(V):
    return .1 * (V + 40)/(1 - np.exp(-.1 * (V + 40)))
def alpha_h(V):
    return .07 * np.exp(-.05 * (V + 65))
def beta_n(V):
    return .125 * np.exp(-.0125 * (V + 65))
def beta_m(V):
    return 4 * np.exp(-.0556 * (V + 65))
def beta_h(V):
    return 1/(1 + np.exp(-.1 * (V + 35)))
def dndt(n, V):
    return alpha_n(V)*(1-n) - beta_n(V)*n
def dmdt(m, V):
    return alpha_m(V)*(1-m) - beta_m(V) *m
def dhdt(h, V):
    return alpha_h(V)*(1-h) - beta_h(V) * h
def tau_n(V):
    return 1/(alpha_n(V) + beta_n(V))
def tau_m(V):
    return 1/(alpha_m(V) + beta_m(V))
def tau_h(V):
    return 1/(alpha_h(V) + beta_h(V))
def n_ss(V):
    return alpha_n(V) * tau_n(V)
def m_ss(V):
    return alpha_m(V) * tau_m(V)
def h_ss(V):
    return alpha_h(V) * tau_h(V)

def i_m(V, n, m, h):
    return g_L*(V - E_L) +g_K * n ** 4 * (V - E_K) + g_Na * m ** 3 * h * (V - E_Na)

#Defining a variant of i_m that makes
def i_m_persistent(V, n, m, h):
    return g_L*(V - E_L) +g_K * n ** 4 * (V - E_K) + g_Na * m ** 4 * (V - E_Na)

def dVdt(membrane_curr, I_e):
    return (1/c_m) * (-membrane_curr + I_e / A)

#Runs and plots all figures, with a boolean option of allowing for persistent sodium channels
def runAndPlot(fig_start, extra_text, persistent_sodium):
    times = np.linspace(0, t_final, int(t_final / dt) + 1)
    n = np.zeros(len(times))
    m = np.zeros(len(times))
    h = np.zeros(len(times))
    V = np.zeros(len(times))
    I_e = np.zeros(len(times))
    i_mem = np.zeros(len(times))

    t_index_start_curr = 5000
    for i in range(t_index_start_curr, len(times)):
        I_e[i] = 20 * 10 ** -3

    n[0], m[0], h[0], V[0] = n_ss(E_rest), m_ss(E_rest), h_ss(E_rest), E_rest
    i_mem[0] = i_m(V[0], n[0], m[0], h[0])

    if persistent_sodium:
        i_mem_used = i_m_persistent
    else:
        i_mem_used = i_m
    for i in range(1, len(times)):
        i_mem[i] = i_mem_used(V[i - 1], n[i - 1], m[i - 1], h[i - 1])
        V[i] = V[i - 1] + dt * dVdt(i_mem[i - 1], I_e[i - 1])
        n[i] = n[i - 1] + dt * dndt(n[i - 1], V[i - 1])
        m[i] = m[i - 1] + dt * dmdt(m[i - 1], V[i - 1])
        h[i] = h[i - 1] + dt * dhdt(h[i - 1], V[i - 1])

    print(V)
    print(len(V))
    voltage_timeplot = plt.figure(fig_start)
    plt.title("Voltage over time "+extra_text)
    plt.xlabel("time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.plot(times, V)

    membcurr_timeplot = plt.figure(fig_start + 1)
    plt.title("Membrane current over Time " +extra_text)
    plt.xlabel("time (ms)")
    plt.ylabel("specific membrane current (microAmperes/mm^2)")
    # plt.ylim([-5, 5])
    plt.plot(times, i_mem)

    probabilities_timeplot = plt.figure(fig_start + 2)
    plt.title("Activation Rates over Time "+extra_text)
    plt.xlabel("time (ms)")
    plt.ylabel("Activation rate")
    plt.ylim([0, 1])
    plt.plot(times, n, label="n", color="black")
    plt.plot(times, m, label="m", color="blue")
    plt.plot(times, h, label="h", color="red")
    plt.legend()

voltage_vals = np.linspace(-100, 0)
alpha_vals = alpha_n(voltage_vals)
beta_vals = beta_n(voltage_vals)
n_ss_vals = n_ss(voltage_vals)
m_ss_vals = m_ss(voltage_vals)
h_ss_vals = h_ss(voltage_vals)
tau_n_vals = tau_n(voltage_vals)
tau_m_vals = tau_m(voltage_vals)
tau_h_vals = tau_h(voltage_vals)

alphabeta_plot = plt.figure(1)
plt.title("alpha, beta  vs Voltage")
plt.xlabel("V (mV)")
plt.ylabel("alpha_n or beta_n (ms ^ -1)")
plt.plot(voltage_vals, alpha_vals, color = "blue", label = "alpha")
plt.plot(voltage_vals, beta_vals, color = "black", label = "beta")
plt.legend()

n_ss_plot = plt.figure(2)
plt.title("Steady state n activation rate vs Voltage")
plt.xlabel("V (mV)")
plt.ylabel("n_inf")
plt.plot(voltage_vals, n_ss_vals)

tau_n_plot = plt.figure(3)
plt.title("n Time constant vs Voltage")
plt.xlabel("V (mV)")
plt.ylabel("tau_n (ms)")
plt.plot(voltage_vals, tau_n_vals)

activationrate_plot = plt.figure(4)
plt.title("Steady state activation rates vs Voltage")
plt.xlabel("V (mV)")
plt.ylabel("steady-state activation rate")
plt.plot(voltage_vals, n_ss_vals, label = "n_ss", color = "black")
plt.plot(voltage_vals, m_ss_vals, label = "m_ss", color = "blue")
plt.plot(voltage_vals, h_ss_vals, label = "h_ss", color = "red")
plt.legend()

tau_plot = plt.figure(5)
plt.title("Time constant vs Voltage")
plt.xlabel("V (mV)")
plt.ylabel("Time constant (ms)")
plt.plot(voltage_vals, tau_n_vals, label = "tau_n", color = "black")
plt.plot(voltage_vals, tau_m_vals, label = "tau_m", color = "blue")
plt.plot(voltage_vals, tau_h_vals, label = "tau_h", color = "red")
plt.legend()

plt.show()
#numerical simulation

runAndPlot(6, "", False)

#simulations with toxins below

g_Na /= 10

runAndPlot(9, "with blocked sodium channels", False)

#Resetting g_Na and poisoning potassium channels
g_Na *= 10
g_K /= 10

runAndPlot(12, "with blocked potassium channels", False)

#Resetting everything
g_K *= 10

#persistent sodium channels
runAndPlot(15, "with persistent sodium channels", True)

plt.show()