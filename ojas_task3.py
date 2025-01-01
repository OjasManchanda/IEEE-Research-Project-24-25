import numpy as np

def get_matrix_input(name):
    
    print(f"Enter the dimensions of matrix {name} (rows, columns):")
    rows, cols = map(int, input().split())
    
    print(f"Enter the elements of matrix {name} row by row :")
    matrix = []
    for _ in range(rows):
        row = list(map(int, input().split()))
        if len(row) != cols:
            raise ValueError(f"Each row must have exactly {cols} columns.")
        matrix.append(row)
    
    return np.array(matrix)

print("Matrix Multiplication: A x B")
A = get_matrix_input("A")
B = get_matrix_input("B")


if A.shape[1] != B.shape[0]:
    raise ValueError("Number of columns in A must equal the number of rows in B.")

rows_A, cols_A = A.shape
rows_B, cols_B = B.shape

C = np.zeros((rows_A, cols_B))

for i in range(rows_A):
    for j in range(cols_B):
        C[i, j] = sum(A[i, k] * B[k, j] for k in range(cols_A))

print("\n")
print("Matrix A:")
print(A)
print("\n")

print("Matrix B:")
print(B)
print("\n")

print("Result of Matrix Multiplication (A x B):")
print(C)

