import numpy as np
import stumpy
import time
import random


def create_step_pattern(n1):
    pattern = [1] * n1 + [8] * n1 + [1] * n1
    return pattern

n1 = 100
num_repetitions = 300
noise_std = 0.5
pattern = create_step_pattern(n1)

T = np.tile(pattern, num_repetitions).astype(np.float64)
print("Length T      :", len(T))

np.random.seed(42)
T_noisy = T + np.random.normal(loc=0.0, scale=noise_std, size=len(T))
print("Length T noise:", len(T_noisy))

T = T_noisy


W = 10
k_max = 100
Eu_Thresh = 0.5
match_thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 100, 1000]


print('\n')

print(f"W: {W}, k_max: {k_max}, Eu_Thresh: {Eu_Thresh}")

time_list = []

# Warm-up
_ = stumpy.stump(T_A=T, m=W, normalize=False, k=k_max, eu_thres=Eu_Thresh)

num_trials = 5
for i in range(num_trials):
    start = time.perf_counter()
    profile = stumpy.stump(T_A=T, m=W, normalize=False, k=k_max, eu_thres=Eu_Thresh)
    end = time.perf_counter()
    total_time = round(end- start, 2)
    time_list.append(total_time)
    # print(f"Time: {total_time} seconds")

# Print timing results
print(f"Avg Time:    {round(np.mean(time_list), 2)} sec")
print(f"Median Time: {round(np.median(time_list), 2)} sec")
print(f"Min Time:    {round(np.min(time_list), 2)} sec")

for time in time_list:
    print(f"Time: {time} seconds")

# print(profile)
print("profile shape:", profile.shape)
distances = profile[:, :k_max] 
print("distance shape:", distances.shape)

for threshold in match_thresholds:
    count = np.sum(distances <= threshold)
    print(f"Valid matches (< {threshold}): {count}")

