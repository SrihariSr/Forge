import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import math
import random
from forge import Tensor
from forge.dtype import float64

def standard_normals(count, seed=42) -> Tensor:
    """
    Draws `count` samples from a normal distribution with mean 0 and standard deviation 1
    using the Box-Muller transform.
    """
    rand = random.Random(seed)
    values = []

    while len(values) < count:
        u1 = rand.random()
        u2 = rand.random()

        # Draw again if an invalid number is picked
        while u1 <= 0.0:
            u1 = rand.random()
        
        radius = math.sqrt(-2.0 * math.log(u1))
        angle = 2.0 * math.pi * u2
        
        values.append(radius * math.cos(angle))
        
        if len(values) < count:
            values.append(radius * math.sin(angle))

    return Tensor(values, dtype=float64)

def _normal_cdf(x) -> float:
    """
    Returns the probability that a draw from the normal distribution is less than `x`.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black_scholes_call(S0, K, r, sigma, T) -> float:
    """
    The exact price of an ordinary call option using the Black-Scholes formula.
    """
    d1 = (math.log(S0 / K) + (r + sigma * sigma / 2.0) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
 
    return S0 * _normal_cdf(d1) - K * math.exp(-r * T) * _normal_cdf(d2)

def monte_carlo_call(S0, K, r, sigma, T, paths, seed=42, antithetic=False) -> float:
    """
    Price an ordinary call option by simulating `paths` possible futures.
    """
    drift = (r - sigma * sigma / 2.0) * T
    volatility = sigma * math.sqrt(T)
    discount = math.exp(-r * T)

    if antithetic:
        half = (paths + 1) // 2
        base = standard_normals(half, seed)
        mirrored = [-z for z in base._data]
        draws = Tensor(list(base._data) + mirrored, dtype=float64)
    else:
        draws = standard_normals(paths, seed)
    
    # The simulation
    final_price = (draws * volatility + drift).exp() * S0
    payoff = (final_price - K).relu()

    return discount * payoff.mean()._data[0]

def _max_of(a, b) -> Tensor:
    return a + (b - a).relu()

def simulate_paths(S0, r, sigma, T, paths, steps, seed=None):
    """
    Walk `paths` share prices forward through `steps` equal slices of time,
    recording the data about each journey.
    Each path runs in parallel so the loop is over time not over paths.
    """
    dt = T / steps # length of a slice of time
    drift = (r - sigma**2 / 2) * dt # predictable move over time
    volatility = sigma * math.sqrt(dt) # random move per slice

    # Current price
    current = Tensor([S0] * paths, dtype=float64)
    
    total = Tensor([0.0] * paths, dtype=float64)
    log_sum = Tensor([0.0] * paths, dtype=float64)
    highest = Tensor([S0] * paths, dtype=float64)

    for step in range(steps):
        draws = standard_normals(paths, None if seed is None else seed*67 + step)

        current = current * (draws*volatility + drift).exp()

        total += current
        log_sum += current.log()
        highest = _max_of(highest, current)

    return current, total, log_sum, highest

def asian_call(S0, K, r, sigma, T, paths, steps=100, seed=66, geometric=False) -> float:
    """
    Prices an asian call with geometric pricing to validate simulation.
    """
    _, total, log_sum, _ = simulate_paths(S0, r, sigma, T, paths, steps, seed)

    if geometric:
        average = (log_sum / steps).exp()
    else:
        average = total / steps
    
    payoff = (average - K).relu()
    return math.exp(-r * T) * payoff.mean()._data[0]

def barrier_call(S0, K, B, r, sigma, T, paths, steps=100, seed=66) -> float:
    """
    An ordinary call that is cancelled if the share ever rises above
    the barrier `B` before expiry.
    """

    final, _, _, highest = simulate_paths(S0, r, sigma, T, paths, steps, seed)
    
    payoff = (final - K).relu()

    # `breached` has a binary value, 0 = under the barrier, 1 = crossed the barrier
    breached = ((highest - B).relu() * 1e13).clamp(0.0, 1.0)

    # complimenting `breached`
    alive = breached * -1.0 + 1.0

    return math.exp(-r * T) * (payoff * alive).mean()._data[0]

def geometric_asian_closed_form(S0, K, r, sigma, T, steps) -> float:
    """
    The exact price of a geometric-average asian call observed `steps` times.
    """
    s = steps

    # Calculate the mean and standard deviation of the log-geometric average
    geometric_mean_log = math.log(S0) + (r - sigma**2 / 2.0) * T * (s + 1) / (2.0 * s)
    geometric_std_log = math.sqrt(sigma**2 * T * (s + 1) * (2*s + 1) / (6.0 * s**2))

    # Plug the adjusted geometric parameters into the standard d1/d2 formulas
    d1 = (geometric_mean_log - math.log(K) + geometric_std_log**2) / geometric_std_log
    d2 = d1 - geometric_std_log

    # Discounted Black-Scholes style payoff
    expected_geometric_S = math.exp(geometric_mean_log + geometric_std_log**2 / 2.0)
    
    return math.exp(-r * T) * (expected_geometric_S * _normal_cdf(d1) - K * _normal_cdf(d2))
