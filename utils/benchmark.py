import time
import numpy as np
import pandas as pd


def benchmark(func, n_repeats=5, *args, **kwargs):
    """
    Benchmark a single function by running it multiple times and averaging the execution time.

    Parameters:
    - func: Function to benchmark.
    - n_repeats: Number of times to run the function.
    - *args: Positional arguments for the function.
    - **kwargs: Keyword arguments for the function.

    Returns:
    - Average execution time over the repetitions.
    - Standard deviation of execution times.
    """
    times = []
    for _ in range(n_repeats):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times), np.std(times)


def matrix_multiplication(size=3000):
    """
    Perform matrix multiplication on large random matrices.
    """
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    return np.dot(a, b)


def sorting(array_size=10**7):
    """
    Perform sorting on a large random array.
    """
    data = np.random.rand(array_size)
    return np.sort(data)


def elementwise_operations(size=10**7):
    """
    Perform element-wise operations on a large array.
    """
    a = np.random.rand(size)
    b = np.random.rand(size)
    return a + b, a * b, np.sqrt(a), np.exp(b)


def simulate_workload(iterations=10**8):
    """
    Simulate a computational workload by performing iterative computations.
    """
    result = 0
    for i in range(iterations):
        result += (i % 7) * (i % 5)
    return result


def eigen_decomposition(size=2000):
    """
    Perform eigenvalue decomposition on a large random matrix.
    """
    matrix = np.random.rand(size, size)
    return np.linalg.eig(matrix)


def fft_computation(size=10**7):
    """
    Perform Fast Fourier Transform (FFT) on a large random array.
    """
    data = np.random.rand(size)
    return np.fft.fft(data)


if __name__ == "__main__":
    # Benchmark parameters
    n_repeats = 7  # Number of repetitions for each function

    # List of functions to benchmark and their parameters
    benchmark_tasks = [
        {"name": "Matrix Multiplication", "func": matrix_multiplication, "kwargs": {"size": 3000}},
        {"name": "Sorting", "func": sorting, "kwargs": {"array_size": 10**7}},
        {"name": "Elementwise Operations", "func": elementwise_operations, "kwargs": {"size": 10**7}},
        {"name": "Simulated Workload", "func": simulate_workload, "kwargs": {"iterations": 10**8}},
        {"name": "Eigen Decomposition", "func": eigen_decomposition, "kwargs": {"size": 2000}},
        {"name": "FFT Computation", "func": fft_computation, "kwargs": {"size": 10**7}},
    ]

    # Store results in a DataFrame
    results = []
    for task in benchmark_tasks:
        print(f"Running benchmark: {task['name']}...")
        avg_time, std_time = benchmark(
            task["func"], n_repeats=n_repeats, **task["kwargs"]
        )
        results.append({
            "Functionality": task["name"],
            "Average Time (s)": avg_time,
            "Standard Deviation (s)": std_time,
        })
        print(f"  Avg: {avg_time:.6f}s, Std: {std_time:.6f}s\n")

    # Display results as a DataFrame
    results_df = pd.DataFrame(results)
    print("Benchmark Results:")
    print(results_df)

    # Optional: Save results to a CSV file
    # results_df.to_csv("benchmark_results_heavy.csv", index=False)