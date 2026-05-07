"""
Request scheduling helpers for single-turn benchmarks.

The runner applies an asyncio semaphore after these timestamps are generated,
so `concurrency` is always a max-in-flight cap. The paper-facing sweeps use the
default "steady" pattern: every request is scheduled at t=0 and the semaphore
keeps up to C requests active, launching queued requests as slots free. This is
closed-loop concurrency with zero client think time.

"poisson" and "ramp" schedule requests by absolute arrival time, but they are
still capped by the same semaphore in the runner. They are useful sensitivity
tools, not the default paper methodology and not a pure open-loop generator
under overload.
"""

import random
import math


def steady_arrivals(num_requests: int, concurrency: int) -> list[float]:
    """
    Closed-loop max-in-flight scheduling.

    All requests are eligible at t=0. The runner's semaphore enforces the
    requested concurrency, so new queued requests start as soon as earlier
    requests finish. `concurrency` is accepted here for factory symmetry; the
    cap is enforced in the runner.
    """
    times = []
    # All requests at t=0 (fire-and-forget concurrency control via semaphore)
    for i in range(num_requests):
        times.append(0.0)
    return times


def poisson_arrivals(num_requests: int, target_rate: float, seed: int = 42) -> list[float]:
    """
    Poisson process: inter-arrival times are exponentially distributed.

    These timestamps are still subject to the runner's max-in-flight semaphore,
    so this is a scheduled-arrival mode with backpressure, not a pure open-loop
    load generator when the server cannot keep up.

    Args:
        num_requests: total number of requests to schedule
        target_rate: average requests per second
        seed: random seed for reproducibility

    Returns:
        List of cumulative arrival times (seconds from t=0)
    """
    rng = random.Random(seed)
    times = []
    t = 0.0
    for _ in range(num_requests):
        # Exponential inter-arrival time: mean = 1/rate
        inter_arrival = -math.log(1.0 - rng.random()) / target_rate
        t += inter_arrival
        times.append(t)
    return times


def ramp_arrivals(num_requests: int, start_rate: float, end_rate: float, seed: int = 42) -> list[float]:
    """
    Linearly increasing arrival rate from start_rate to end_rate.

    Like poisson_arrivals, ramp timestamps are capped by the runner's semaphore.
    Use this as a sensitivity probe rather than the paper-facing saturation
    methodology.

    Returns list of cumulative arrival times.
    """
    rng = random.Random(seed)
    times = []
    t = 0.0
    for i in range(num_requests):
        fraction = i / max(num_requests - 1, 1)
        rate = start_rate + fraction * (end_rate - start_rate)
        inter_arrival = -math.log(1.0 - rng.random()) / rate
        t += inter_arrival
        times.append(t)
    return times


def make_arrival_times(
    pattern: str,
    num_requests: int,
    concurrency: int = 10,
    target_rate: float = 10.0,
    seed: int = 42,
) -> list[float]:
    """
    Create single-turn request schedule timestamps.

    Args:
        pattern: "steady", "poisson", or "ramp"
        num_requests: number of requests
        concurrency: max-in-flight cap applied by the runner for all patterns
        target_rate: requests/sec for "poisson" and "ramp"
        seed: random seed
    """
    if pattern == "steady":
        return steady_arrivals(num_requests, concurrency)
    elif pattern == "poisson":
        return poisson_arrivals(num_requests, target_rate, seed)
    elif pattern == "ramp":
        return ramp_arrivals(num_requests, target_rate * 0.1, target_rate, seed)
    else:
        raise ValueError(f"Unknown arrival pattern: '{pattern}'. Use 'steady', 'poisson', or 'ramp'.")
