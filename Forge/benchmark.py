import time


class BenchmarkResult:
    # Stores the result of a single benchmark

    def __init__(self, name, my_time, pytorch_time=None):
        self.name = name
        self.my_time = my_time
        self.pytorch_time = pytorch_time

    @property
    def speedup(self):
        if self.pytorch_time is None or self.pytorch_time == 0:
            return None
        return self.my_time / self.pytorch_time

    def __repr__(self):
        line = f"{self.name:40s} | Forge: {self.my_time * 1000:10.3f}ms"
        if self.pytorch_time is not None:
            line += f" | PyTorch: {self.pytorch_time * 1000:10.3f}ms"
            line += f" | Ratio: {self.speedup:8.1f}x"
        return line


def time_fn(fn, warmup=2, runs=10):
    # Warmup runs
    for _ in range(warmup):
        fn()

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append(end - start)

    # Return median
    times.sort()
    mid = len(times) // 2
    return times[mid]

def benchmark_matmul(sizes=None):
    # Benchmark matrix multiplication at different sizes
    from Forge.tensor import Tensor
    import random

    if sizes is None:
        sizes = [(16, 16), (32, 32), (64, 64), (128, 128)]

    results = []

    for m, n in sizes:
        # Create random data as nested lists
        data_a = [[random.random() for _ in range(n)] for _ in range(m)]
        data_b = [[random.random() for _ in range(m)] for _ in range(n)]

        # My library
        a = Tensor(data_a)
        b = Tensor(data_b)
        my_time = time_fn(lambda: a @ b, warmup=1, runs=5)

        # PyTorch comparison
        pt_time = None
        try:
            import torch
            ta = torch.tensor(data_a)
            tb = torch.tensor(data_b)
            pt_time = time_fn(lambda: ta @ tb, warmup=1, runs=5)
        except ImportError:
            pass

        results.append(BenchmarkResult(
            f"Matmul ({m}x{n}) @ ({n}x{m})",
            my_time,
            pt_time
        ))

    return results


def benchmark_elementwise(size=None):
    # Benchmark elementwise operations
    from Forge.tensor import Tensor
    import random

    if size is None:
        size = 100000

    data_a = [random.random() for _ in range(size)]
    data_b = [random.random() for _ in range(size)]

    a = Tensor(data_a)
    b = Tensor(data_b)

    results = []

    # Addition
    my_time = time_fn(lambda: a + b)
    pt_time = None
    try:
        import torch
        ta = torch.tensor(data_a)
        tb = torch.tensor(data_b)
        pt_time = time_fn(lambda: ta + tb)
    except ImportError:
        pass
    results.append(BenchmarkResult(f"Add ({size} elements)", my_time, pt_time))

    # Multiplication
    my_time = time_fn(lambda: a * b)
    try:
        import torch
        pt_time = time_fn(lambda: ta * tb)
    except ImportError:
        pt_time = None
    results.append(BenchmarkResult(f"Mul ({size} elements)", my_time, pt_time))

    # Power
    my_time = time_fn(lambda: a ** 2)
    try:
        import torch
        pt_time = time_fn(lambda: ta ** 2)
    except ImportError:
        pt_time = None
    results.append(BenchmarkResult(f"Pow ({size} elements)", my_time, pt_time))

    return results


def benchmark_activations(size=None):
    # Benchmark activation functions
    from Forge.tensor import Tensor
    import random

    if size is None:
        size = 10000

    data = [random.uniform(-5, 5) for _ in range(size)]
    a = Tensor(data)

    results = []

    # ReLU
    my_time = time_fn(lambda: a.relu())
    pt_time = None
    try:
        import torch
        ta = torch.tensor(data)
        pt_time = time_fn(lambda: torch.relu(ta))
    except ImportError:
        pass
    results.append(BenchmarkResult(f"ReLU ({size} elements)", my_time, pt_time))

    # Sigmoid
    my_time = time_fn(lambda: a.sigmoid())
    try:
        import torch
        pt_time = time_fn(lambda: torch.sigmoid(ta))
    except ImportError:
        pt_time = None
    results.append(BenchmarkResult(f"Sigmoid ({size} elements)", my_time, pt_time))

    # Tanh
    my_time = time_fn(lambda: a.tanh())
    try:
        import torch
        pt_time = time_fn(lambda: torch.tanh(ta))
    except ImportError:
        pt_time = None
    results.append(BenchmarkResult(f"Tanh ({size} elements)", my_time, pt_time))

    return results


def benchmark_training(epochs=100):
    # Benchmark a full training loop
    from Forge.tensor import Tensor
    from NeuralNetwork.layers import Linear, ReLULayer, Sequential
    from NeuralNetwork.losses import MSELoss
    from NeuralNetwork.optim import SGD

    X_data = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    Y_data = [[0.0], [1.0], [1.0], [0.0]]

    def my_train():
        X = Tensor(X_data)
        Y = Tensor(Y_data)
        model = Sequential(
            Linear(2, 8),
            ReLULayer(),
            Linear(8, 1),
        )
        criterion = MSELoss()
        optimizer = SGD(model.parameters(), lr=0.1)

        for _ in range(epochs):
            pred = model(X)
            loss = criterion(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    my_time = time_fn(my_train, warmup=1, runs=3)

    # PyTorch comparison
    pt_time = None
    try:
        import torch
        import torch.nn as pt_nn

        def pt_train():
            X = torch.tensor(X_data)
            Y = torch.tensor(Y_data)
            model = pt_nn.Sequential(
                pt_nn.Linear(2, 8),
                pt_nn.ReLU(),
                pt_nn.Linear(8, 1),
            )
            criterion = pt_nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

            for _ in range(epochs):
                pred = model(X)
                loss = criterion(pred, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        pt_time = time_fn(pt_train, warmup=1, runs=3)
    except ImportError:
        pass

    return [BenchmarkResult(f"Training loop ({epochs} epochs)", my_time, pt_time)]

def run_all_benchmarks():
    # Run all benchmarks and print a report
    print("-" * 80)
    print("BENCHMARK REPORT")
    print("-" * 80)

    all_results = []

    print("\nMatrix Multiplication")
    results = benchmark_matmul()
    all_results.extend(results)
    for r in results:
        print(r)

    print("\nElementwise Operations")
    results = benchmark_elementwise()
    all_results.extend(results)
    for r in results:
        print(r)

    print("\nActivation Functions")
    results = benchmark_activations()
    all_results.extend(results)
    for r in results:
        print(r)

    print("\nTraining")
    results = benchmark_training()
    all_results.extend(results)
    for r in results:
        print(r)

    # Summary
    print("-" * 80)
    print("SUMMARY")
    print("-" * 80)

    ratios = [r.speedup for r in all_results if r.speedup is not None]
    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        print(f"Average slowdown vs PyTorch: {avg_ratio:.1f}x")
        print(f"Best case: {min_ratio:.1f}x")
        print(f"Worst case: {max_ratio:.1f}x")
    else:
        print("No PyTorch comparison available (torch not installed)")
