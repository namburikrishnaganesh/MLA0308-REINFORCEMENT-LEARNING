import numpy as np
n_rows, n_cols = 6, 4
actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
state_values = np.zeros((n_rows, n_cols))
rewards = np.zeros((n_rows, n_cols))
rewards[5, 3] = 1 
gamma = 0.9
num_iterations = 100
for _ in range(num_iterations):
    new_state_values = np.copy(state_values)
    for i in range(n_rows):
        for j in range(n_cols):
            if rewards[i, j] != 0:
                continue
            q_values = []
            for action in actions:
                next_i, next_j = i + action[0], j + action[1]
                if 0 <= next_i < n_rows and 0 <= next_j < n_cols:
                    q_values.append(state_values[next_i, next_j])
            if q_values:
                new_state_values[i, j] = max(q_values) * gamma
    state_values = new_state_values
print(state_values)
