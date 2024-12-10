import numpy as np
from tqdm import tqdm

def forward(obs_seq, A, B, pi):
    T = len(obs_seq)
    N = len(A)
    alpha = np.zeros((T, N))
    
    # init
    alpha[0, :] = pi * B[:, obs_seq[0]]
    
    # recursion
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t - 1, :] * A[:, j]) * B[j, obs_seq[t]]
    
    # prob
    prob = np.sum(alpha[-1, :])
    return alpha, prob

def backward(obs_seq, A, B):
    T = len(obs_seq)
    N = len(A)
    beta = np.zeros((T, N))
    
    # init
    beta[-1, :] = 1
    
    # recursion
    for t in range(T - 2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(A[i, :] * B[:, obs_seq[t + 1]] * beta[t + 1, :])
    
    return beta

def baum_welch(obs_sequences, N, M, A=None, B=None, pi=None, max_iter=100, tol=1e-4):
    """
    The multi-sequence version of the Baum-Welch algorithm supports manual input of initialization parameters.
    :param obs_sequences: Multiple observation sequences (list, each sequence is a list of integers)
    :param N: number of hidden states
    :param M: Number of observed symbol types
    :param A: initial transition matrix (optional)
    :param B: initial emission matrix (optional)
    :param pi: initial state distribution (optional)
    :param max_iter: Maximum number of iterations
    :param tol: convergence judgment threshold
    :return: updated A, B, pi
    """
    # init randomly if not given
    if A is None:
        A = np.random.rand(N, N)
        A /= A.sum(axis=1, keepdims=True)
    
    if B is None:
        B = np.random.rand(N, M)
        B /= B.sum(axis=1, keepdims=True)
    
    if pi is None:
        pi = np.random.rand(N)
        pi /= pi.sum()
    
    prev_prob = -np.inf
    
    for iteration in tqdm(range(max_iter), desc="Training iter : "):
        # init adder
        gamma_sum = np.zeros((N,))
        xi_sum = np.zeros((N, N))
        B_num = np.zeros((N, M))
        B_denom = np.zeros((N,))
        pi_sum = np.zeros((N,))
        
        total_prob = 0  # the total log-likelihood
        
        # Calculate the expected value for each obs_seq
        for obs_seq in obs_sequences:
            T = len(obs_seq)
            
            # forward and backward algorithm
            alpha, prob = forward(obs_seq, A, B, pi)
            beta = backward(obs_seq, A, B)
            total_prob += np.log(prob)
            
            # calculate gamma and xi
            gamma = np.zeros((T, N))
            xi = np.zeros((T - 1, N, N))
            
            for t in range(T - 1):
                denom = np.sum(alpha[t, :] * np.sum(A * B[:, obs_seq[t + 1]] * beta[t + 1, :], axis=1))
                for i in range(N):
                    gamma[t, i] = (alpha[t, i] * beta[t, i]) / np.sum(alpha[t, :] * beta[t, :])
                    for j in range(N):
                        xi[t, i, j] = (alpha[t, i] * A[i, j] * B[j, obs_seq[t + 1]] * beta[t + 1, j]) / denom
            
            gamma[-1, :] = alpha[-1, :] * beta[-1, :] / np.sum(alpha[-1, :] * beta[-1, :])
            
            # add to pi_sum
            pi_sum += gamma[0, :]
            
            # add to xi_sum
            xi_sum += np.sum(xi, axis=0)
            
            # add to B_num and B_denom
            for t in range(T):
                B_num[:, obs_seq[t]] += gamma[t, :]
                B_denom += gamma[t, :]
            
            gamma_sum += gamma[:-1, :].sum(axis=0)
        
        # update param
        pi = pi_sum / len(obs_sequences)
        A = xi_sum / gamma_sum[:, None]  
        B = B_num / B_denom[:, None]
        
        # check for convergence
        if abs(total_prob - prev_prob) < tol:
            print(f"Converged on iteration {iteration + 1}. ")
            break
        prev_prob = total_prob
    
    return A, B, pi

def train_omm(sequences, num_states):
    
    A = np.zeros((num_states, num_states))
    pi = np.zeros(num_states)

    for seq in sequences:
        first_state = seq[0]
        pi[first_state] += 1

    for seq in sequences:
        for i in range(len(seq) - 1):
            current_state = seq[i]
            next_state = seq[i + 1]
            A[current_state, next_state] += 1

    pi /= np.sum(pi)  
    A /= A.sum(axis=1, keepdims=True)  

    return A, pi

if __name__ == "__main__":
    
    # Initialize HMM parameters
    states = ['S1', 'S2', 'S3']
    symbols = ['A', 'B', 'C']
    state_count = len(states)
    symbol_count = len(symbols)

    # Initial probabilities (uniform)
    pi_init = np.array([1/3, 1/3, 1/3])

    # Transition probabilities (from the diagram)
    A_init = np.array([
        [0.34, 0.33, 0.33],  # S1 -> S1, S2, S3
        [0.33, 0.34, 0.33],  # S2 -> S1, S2, S3
        [0.33, 0.33, 0.34]   # S3 -> S1, S2, S3
    ])
    
    # Emission probabilities (from the diagram)
    B_init = np.array([
        [0.34, 0.33, 0.33],  # S1 emits A, B, C
        [0.33, 0.34, 0.33],  # S2 emits A, B, C
        [0.33, 0.33, 0.34]   # S3 emits A, B, C
    ])

    obs_sequences = [
        [0, 1, 2, 1, 0],  # 序列 1
        [2, 2, 1, 0, 1]   # 序列 2
    ]
    
    # Baum-Welch training
    A, B, pi = baum_welch(obs_sequences, state_count, symbol_count, A=A_init, B=B_init, pi=pi_init, max_iter=50, tol=0)

    # 輸出結果
    print("更新後的轉移矩陣 A:")
    print(A)
    print("\n更新後的發射概率矩陣 B:")
    print(B)
    print("\n更新後的初始狀態分佈 pi:")
    print(pi)
    
    for i in range(3):
        print(np.sum(A[i, :]))
        print(np.sum(B[i, :]))