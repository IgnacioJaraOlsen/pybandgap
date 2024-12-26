import numpy as np

def P_norm(matrix, P):
    # Raise each element to the power P
    powered_matrix = np.power(matrix, P)
    
    # Sum all elements in the powered matrix
    summed_value = np.sum(powered_matrix)
    
    # Raise the summed value to the power of 1/P
    result = np.power(summed_value, 1 / P)
    
    return result

def objective_p_norm_diff(bands, n, P):
    min_w_np1 = P_norm(bands[:,:n+1], -P)
    max_w_n = P_norm(bands[:,n+1:], P)
    
    delta = min_w_np1 - max_w_n
    
    return delta

def objetive_max_min_diff(bands, n):
    max = np.max(bands[:, n - 1])
    min = np.min(bands[:, n])
    delta = min - max 
    return delta