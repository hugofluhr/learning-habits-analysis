import time
import numpy as np

# Single-core matrix multiplication test
size = 2000
a = np.random.rand(size, size)
b = np.random.rand(size, size)

start_time = time.time()
result = np.dot(a, b)  # Matrix multiplication
end_time = time.time()

print(f"Single-core test: Matrix size {size}x{size} took {end_time - start_time:.2f} seconds.")