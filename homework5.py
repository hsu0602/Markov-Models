from train import baum_welch, train_omm
from decode import viterbi_algorithm, forward_algorithm
import pandas as pd
import numpy as np
from tabulate import tabulate

dataset1 = pd.read_csv("./TrainSet1.csv") 
dataset2 = pd.read_csv("./TrainSet2.csv")
datasets = [dataset1, dataset2]

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

def dataProcess(dataset):
    obs_sequences = []
    for data in dataset['data']:
        symbol_to_index = {symbol: index for index, symbol in enumerate(symbols)}
        obs_sequence = [symbol_to_index[char] for char in data]
        obs_sequences.append(obs_sequence)
    return obs_sequences


def P1P2template(itr, obs_sequences):
    A, B, pi = baum_welch(obs_sequences, state_count, symbol_count, A=A_init, B=B_init, pi=pi_init, max_iter=itr, tol=0)
    print("")
    print(f"A(after itr {itr}) = \n")
    print(tabulate(A, tablefmt="fancy_grid")) 
    print(f"B(after itr {itr}) = \n")
    print(tabulate(B, tablefmt="fancy_grid"))
    print(f"pi(after itr {itr}) = \n")
    print(tabulate([pi], tablefmt="fancy_grid"))
    print("")
    
    print(f"{f'Sequence(itr {itr})'.ljust(20)} {'Best state Sequence'.ljust(45)} {'Probability'}")
    print("=" * 85)
    idxx = 0
    for obs_sequence in obs_sequences:
        idxx = idxx + 1
        best_path, best_prob = viterbi_algorithm(obs_sequence, list(range(len(states))), pi, A, B)
        index_to_state = {index: state for index, state in enumerate(states)}
        best_path = [index_to_state[num] for num in best_path]
        
        best_path_str = ", ".join(best_path)  # Convert state list to a string
        print(f"{f'{idxx}'.ljust(20)} {best_path_str.ljust(45)} {best_prob:.10e}")
        
        #print(f"Sequence{idxx}(itr ={itr}): ", end = "")
        #print(f"Best state sequence: {best_path}, \t", end = "")
        #print(f"Probability: {best_prob}")
    print("")
    
def compute_sequence_probability(sequence, A, pi):
    prob = pi[sequence[0]]
    for i in range(len(sequence) - 1):
        prob *= A[sequence[i], sequence[i + 1]]
    return prob

# -------------- P1 and P2 --------------

print("\n -------------- P1 and P2 -------------- \n")

idx = 0
for dataset in datasets:
    
    idx += 1;
    print(f"\n~~~~~~ TrainSet{idx} : ~~~~~~\n")
    obs_sequences = dataProcess(dataset)
    # Baum-Welch training(itr 1)
    P1P2template(1, obs_sequences)
    # Baum-Welch training(itr 50)
    P1P2template(50, obs_sequences)

# -------------- P3 --------------
print("\n -------------- P3 -------------- \n")

test_sequences = ["ABCABCCAB", "AABABCCCCBBB"]
itrs = [1, 50]

for itr in itrs:
    A_list = []
    B_list = []
    pi_list = []
    print(f"Model trained with {itr} itrs: ")
    for dataset in datasets:
        obs_sequences = dataProcess(dataset)
        A, B, pi = baum_welch(obs_sequences, state_count, symbol_count, A=A_init, B=B_init, pi=pi_init, max_iter=itr, tol=0)
        A_list.append(A)
        B_list.append(B)
        pi_list.append(pi)
    
    for test_sequence in test_sequences:
        prob_list = []
        for A, B, pi in zip(A_list, B_list, pi_list):
            symbol_to_index = {symbol: index for index, symbol in enumerate(symbols)}
            test_sequencen = [symbol_to_index[char] for char in test_sequence]
            prob = forward_algorithm(test_sequencen, list(range(len(states))), pi, A, B)
            prob_list.append(prob)
        print("")
        print(f"# Test sequence ({test_sequence}) : ")
        print(f"Belongs to TrainSet {prob_list.index(max(prob_list)) + 1}")
        print(f"Probabilities for Each Training Set: {prob_list}")
        print("")
    print("")

# -------------- P4 --------------
print("\n -------------- P4 -------------- \n")

test_sequences = ["ABCABCCAB", "AABABCCCCBBB"]
idx = 0
A_list = []
pi_list = []
for dataset in datasets:
    idx += 1;
    print(f"\n~~~~~~ TrainSet{idx} : ~~~~~~\n")
    obs_sequences = dataProcess(dataset)
    # Baum-Welch training(itr 1)
    A, pi = train_omm(obs_sequences, state_count)
    A_list.append(A)
    pi_list.append(pi)
    print("OMM Parameters:")
    print("")
    print(f"A = \n")
    print(tabulate(A, tablefmt="fancy_grid")) 
    print(f"pi = \n")
    print(tabulate([pi], tablefmt="fancy_grid"))
    print("")

for test_sequence in test_sequences:
    prob_list = []
    for A, pi in zip(A_list, pi_list):
        symbol_to_index = {symbol: index for index, symbol in enumerate(symbols)}
        test_sequencen = [symbol_to_index[char] for char in test_sequence]
        prob = compute_sequence_probability(test_sequencen, A, pi)
        prob_list.append(prob)
    print(f"# Test sequence ({test_sequence}) : ")
    print(f"Belongs to TrainSet {prob_list.index(max(prob_list)) + 1}")
    print(f"Probabilities for Each Training Set: {prob_list}")
    print("")
print("")



