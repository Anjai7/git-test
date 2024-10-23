import torch
import time

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

# Define a larger matrix size
matrix_size = 20000  # Increasing matrix size to make the computation more intensive

# Create two large random matrices on the GPU
A = torch.randn(matrix_size, matrix_size, device=device)
B = torch.randn(matrix_size, matrix_size, device=device)

# Perform multiple operations to increase GPU load
start_time = time.time()

# Repeat the operations multiple times to increase load
for _ in range(10):  # Increase the loop count to make it more intensive
    # Matrix multiplication
    C = torch.matmul(A, B)

    # Element-wise addition and multiplication
    D = A + B
    E = A * B

    # Matrix exponentiation
    F = torch.pow(C, 2)

    # Perform a final matrix multiplication
    G = torch.matmul(F, D)

# Time the operations
end_time = time.time()

# Print some results (first 5 rows of G)
print("First 5 rows of the final result:")
print(G[:5, :])  # Print first 5 rows

# Check if the result is on GPU
print("Is the result stored on the GPU?:", G.is_cuda)

# Show total execution time
print(f"Total execution time: {end_time - start_time} seconds")
