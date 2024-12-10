import numpy as np

A_init = np.array([
    [0.5, 0.2, 0.3],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
])
B = np.array([
    [0.6, 0.2, 0.2],
    [0.3, 0.4, 0.3],
    [0.2, 0.3, 0.5]
])
pi_init = np.array([0.2, 0.4, 0.4])
obs_seq = [0, 1, 2, 1, 0]
print(B[:, obs_seq[0]])