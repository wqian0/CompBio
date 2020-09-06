import numpy as np
import matplotlib.pyplot as plt
num_steps = 1000 #change to 1000 for part d
def pdf_transform_function(rand_num, mean_rate):
    return -(1/mean_rate) * np.log(rand_num)

def target_pdf(t_w, mean_rate):
    return mean_rate*np.exp(-mean_rate*t_w)

uniform_nums = np.random.random(num_steps)
wait_times = pdf_transform_function(uniform_nums, 1)
step_times = np.zeros(num_steps)
t_tot = 0
for i in range(len(step_times)):
    t_tot += wait_times[i]
    step_times[i] = t_tot

mean_rate = num_steps/t_tot
print("mean rate", mean_rate)

transformed_rand_nums = plt.figure(1)
plt.title("Waiting times")
plt.xlabel("waiting time (s)")
plt.ylabel("count")
A, B, C = plt.hist(wait_times, bins = 25)

transformed_rand_nums_pdf = plt.figure(2)
plt.title("Waiting times PDF")
plt.xlabel("waiting time (s)")
plt.ylabel("Probability density (s^-1)")
plt.step(B[1:], A/((B[1]-B[0])*len(wait_times)), where = 'pre')
plt.plot(np.linspace(0, 5), target_pdf(np.linspace(0, 5), 1))

part_a = plt.figure(3)
plt.title("Simulated molecular motor, 100 steps")
plt.xlabel("time (s)")
plt.ylabel("steps")
plt.xlim(left = 0, right = step_times[len(step_times)-1]+1)
plt.ylim(bottom = 0, top = num_steps + 1)
step_probability_1 = np.linspace(1, num_steps, num = num_steps)
plt.step(step_times, step_probability_1, where = 'mid')

thinned_stepping = np.zeros(num_steps)
total_steps = 0
successful_steps = []
for i in range(len(thinned_stepping)):
    if np.random.random(1) > .6:
        total_steps += 1
        successful_steps.append(i)
    thinned_stepping[i] = total_steps

mean_rate_thin = total_steps/t_tot
print("thinned mean rate", mean_rate_thin)
thinned_wait_times = np.zeros(total_steps-1)
for i in range(len(thinned_wait_times)):
    thinned_wait_times[i] = step_times[successful_steps[i+1]] - step_times[successful_steps[i]]

part_b_hist = plt.figure(10)
plt.title("Thinned waiting times")
plt.xlabel("Waiting time (s)")
plt.ylabel("count")
plt.hist(thinned_wait_times, bins = 25)

part_b = plt.figure(4)
plt.title("Simulated molecular motor, 100 Steps with Thinning")
plt.xlabel("time (s)")
plt.ylabel("steps")
plt.xlim(left = 0, right = step_times[len(step_times)-1]+1)
plt.ylim(bottom = 0, top = total_steps + 1)
plt.step(step_times, thinned_stepping, where = 'mid')

part_c = plt.figure(5)
uniform_nums_fast = np.random.random(3*num_steps)
# 3 times the number of steps in part a will pretty much guarantee that this process will run longer than in part a
wait_times_fast = pdf_transform_function(uniform_nums_fast, .5)
step_times_fast = np.zeros(3*num_steps)

t_tot_fast = 0
for i in range(len(step_times_fast)):
    t_tot_fast += wait_times_fast[i]
    step_times_fast[i] = t_tot_fast

truncated_step_times = []
cutoff_index = 0
for i in range(len(step_times_fast)-1):
    truncated_step_times.append(step_times_fast[i])
    if step_times_fast[i+1] > t_tot:
        break

merged_step_times = np.array(list(step_times) + truncated_step_times)
sorted_merged_times = np.sort(merged_step_times)

mean_rate_merged = len(sorted_merged_times)/ sorted_merged_times[len(sorted_merged_times)-1]
print("mean merged rate", mean_rate_merged)
wait_times_merged = np.zeros(len(sorted_merged_times)-1)
for i in range(len(wait_times)):
    wait_times_merged[i] = sorted_merged_times[i+1] - sorted_merged_times[i]

plt.title("Waiting times for Merged Processes")
plt.xlabel("Waiting time (s)")
plt.ylabel("Count")
plt.hist(wait_times_merged, bins = 25)
plt.show()