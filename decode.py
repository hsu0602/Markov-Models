import numpy as np

def forward_algorithm(observation_seq, states, start_prob, trans_prob, emis_prob):
    num_states = len(states)
    T = len(observation_seq)
    
    # Initialize the forward probabilities matrix
    alpha = np.zeros((T, num_states))
    
    # Step 1: Initialize base cases (t=0)
    for s in range(num_states):
        alpha[0, s] = start_prob[s] * emis_prob[s, observation_seq[0]]
    
    # Step 2: Forward algorithm for t > 0
    for t in range(1, T):
        for s in range(num_states):
            alpha[t, s] = sum(alpha[t-1, s_prev] * trans_prob[s_prev, s] * emis_prob[s, observation_seq[t]]
                              for s_prev in range(num_states))
    
    # Step 3: Termination step: sum probabilities of being in any state at time T-1
    return sum(alpha[T-1, s] for s in range(num_states))

def viterbi_algorithm(observation_seq, states, start_prob, trans_prob, emis_prob):
    num_states = len(states)
    T = len(observation_seq)
    
    # Initialize the dynamic programming tables
    dp = np.zeros((T, num_states))  # dp[t][s] stores the max probability of reaching state s at time t
    path = np.zeros((T, num_states), dtype=int)  # path[t][s] stores the previous state in the optimal path
    
    # Step 1: Initialize base cases (t=0)
    for s in range(num_states):
        dp[0, s] = start_prob[s] * emis_prob[s, observation_seq[0]]
        path[0, s] = 0
    
    # Step 2: Viterbi algorithm for t > 0
    for t in range(1, T):
        for s in range(num_states):
            dp[t, s], path[t, s] = max(
                (dp[t-1, s_prev] * trans_prob[s_prev, s] * emis_prob[s, observation_seq[t]], s_prev)
                for s_prev in range(num_states)
            )
    
    # Step 3: Backtrack to find the optimal path
    best_path = np.zeros(T, dtype=int)
    best_path[T-1] = np.argmax(dp[T-1])
    for t in range(T-2, -1, -1):
        best_path[t] = path[t+1, best_path[t+1]]
    
    # Best path probability
    best_prob = dp[T-1, best_path[T-1]]
    
    return best_path, best_prob