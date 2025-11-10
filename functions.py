
import numpy as np

def matrix_multiplication(A, B):
    if len(A[1]) != len(B[0]):
        raise TypeError(f"Size of {A} != Size of {B}")
    return np.matmul(A, B)

def mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)