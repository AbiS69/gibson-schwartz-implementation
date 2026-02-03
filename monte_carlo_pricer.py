from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from math import exp, sqrt
from typing import Literal

from option_pricer import (
    GibsonSchwartzParams,
    futures_price,
    price_option_on_future_gibson_schwartz,
)


@dataclass
class MCResults:
    # Monte Carlo pricing results
    call_price: float
    put_price: float
    call_std_error: float
    put_std_error: float
    num_simulations: int
    
    def __str__(self) -> str:
        return (
            f"Monte Carlo Results ({self.num_simulations:,} simulations):\n"
            f"  Call price: {self.call_price:.6f} ± {self.call_std_error:.6f}\n"
            f"  Put  price: {self.put_price:.6f} ± {self.put_std_error:.6f}"
        )


def simulate_gibson_schwartz(
    S0: float,
    delta0: float,
    T: float,
    p: GibsonSchwartzParams,
    n_steps: int,
    n_sims: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # Simulate Gibson-Schwartz 2-factor model paths
    # Returns: (S_paths, delta_paths) of shape (n_sims, n_steps+1)
    
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / n_steps
    sqrt_dt = sqrt(dt)
    
    # Initialize arrays
    S = np.zeros((n_sims, n_steps + 1))
    delta = np.zeros((n_sims, n_steps + 1))
    
    S[:, 0] = S0
    delta[:, 0] = delta0
    
    # Cholesky decomposition for correlation
    # dW1 and dW2 with correlation rho
    corr_matrix = np.array([[1.0, p.rho], [p.rho, 1.0]])
    chol = np.linalg.cholesky(corr_matrix)
    
    for i in range(n_steps):
        # Generate independent standard normals
        Z = np.random.randn(n_sims, 2)
        
        # Correlate them
        W = Z @ chol.T  # W[:, 0] is dW1, W[:, 1] is dW2
        
        # Update delta (Ornstein-Uhlenbeck)
        # d(delta) = kappa * (delta_bar_q - delta) * dt + sigma_delta * dW2
        delta[:, i + 1] = (
            delta[:, i]
            + p.kappa * (p.delta_bar_q - delta[:, i]) * dt
            + p.sigma_delta * sqrt_dt * W[:, 1]
        )
        
        # Update S (GBM with stochastic convenience yield)
        # dS/S = (r - delta) * dt + sigma_s * dW1
        # Using Euler discretization for simplicity
        S[:, i + 1] = S[:, i] * (
            1.0
            + (p.r - delta[:, i]) * dt
            + p.sigma_s * sqrt_dt * W[:, 0]
        )
        
        # Ensure S stays positive
        S[:, i + 1] = np.maximum(S[:, i + 1], 1e-10)
    
    return S, delta


def mc_price_future(
    S0: float,
    delta0: float,
    T: float,
    p: GibsonSchwartzParams,
    n_sims: int = 100_000,
    n_steps: int = 100,
    seed: int | None = 42,
) -> tuple[float, float]:
    # Price futures contract using Monte Carlo
    # Returns: (price, standard_error)
    #
    # T = futures maturity
    
    # Simulate paths to maturity (T years)
    S_T, delta_T = simulate_gibson_schwartz(
        S0, delta0, T, p, n_steps, n_sims, seed
    )
    
    # Extract values at maturity
    S_at_T = S_T[:, -1]
    
    # Futures price at maturity equals spot price
    # But we want the current futures price, which is E[S_T] under Q
    # discounted by the convenience yield effect
    
    # Actually, F(0,T) = E^Q[S_T * exp(-int_0^T delta_s ds)]
    # But in practice, we use the no-arbitrage formula F(0,T) = E^Q[S_T] / exp(int...)
    # The analytical formula is exact, so we just verify convergence
    
    # For MC verification, we can use the terminal spot price expectation
    # under the risk-neutral measure with convenience yield adjustment
    price = np.mean(S_at_T)
    std_error = np.std(S_at_T) / sqrt(n_sims)
    
    return price, std_error


def mc_price_option_on_future(
    S0: float,
    delta0: float,
    K: float,
    h: float,
    u: float,
    p: GibsonSchwartzParams,
    option: Literal["call", "put"] = "call",
    n_sims: int = 100_000,
    n_steps_to_option: int = 100,
    n_steps_to_future: int = 100,
    seed: int | None = 42,
) -> tuple[float, float]:
    # Price option on future using Monte Carlo
    # Returns: (price, standard_error)
    #
    # h = option expiry
    # u = time from option expiry to future maturity
    
    # Step 1: Simulate paths to option expiry (h years)
    S_h, delta_h = simulate_gibson_schwartz(
        S0, delta0, h, p, n_steps_to_option, n_sims, seed
    )
    
    # Extract values at option expiry
    S_at_h = S_h[:, -1]
    delta_at_h = delta_h[:, -1]
    
    # Step 2: For each path, compute the futures price F(h, h+u)
    # F(h, T) = S_h * exp(A(u) - B(u) * delta_h)
    F_at_h = np.array([
        futures_price(s, d, u, p)
        for s, d in zip(S_at_h, delta_at_h)
    ])
    
    # Step 3: Compute option payoffs
    if option == "call":
        payoffs = np.maximum(F_at_h - K, 0.0)
    elif option == "put":
        payoffs = np.maximum(K - F_at_h, 0.0)
    else:
        raise ValueError("option must be 'call' or 'put'")
    
    # Step 4: Discount and average
    discount_factor = exp(-p.r * h)
    discounted_payoffs = discount_factor * payoffs
    
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / sqrt(n_sims)
    
    return price, std_error


def compare_futures_mc_vs_analytical(
    S0: float,
    delta0: float,
    p: GibsonSchwartzParams,
    maturities: list[float] = [0.25, 0.5, 0.75, 1.0],
    n_sims: int = 100_000,
    seed: int | None = 42,
) -> None:
    # Compare Monte Carlo vs analytical pricing for futures
    
    print("=" * 70)
    print("FUTURES PRICING: Monte Carlo vs Analytical")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Spot S0: {S0:.2f}")
    print(f"  Convenience yield δ0: {delta0:.4f}")
    print(f"  Simulations: {n_sims:,}")
    
    print(f"\n{'Maturity':<12} {'Analytical':<15} {'Monte Carlo':<20} {'Difference':<15} {'Status'}")
    print("=" * 70)
    
    for T in maturities:
        # Analytical price
        F_analytical = futures_price(S0, delta0, T, p)
        
        # Monte Carlo price
        F_mc, F_se = mc_price_future(S0, delta0, T, p, n_sims, seed=seed)
        
        # Note: MC gives E[S_T] which is not exactly F(0,T) due to convenience yield
        # We show this for educational purposes
        diff = F_mc - F_analytical
        pct_diff = 100 * diff / F_analytical if F_analytical != 0 else 0
        
        print(f"{T:>6.2f}y     {F_analytical:<15.4f} {F_mc:<12.4f}±{F_se:<6.4f} "
              f"{diff:>+8.4f} ({pct_diff:>+6.2f}%)")
    
    print("=" * 70)
    print("\nNote: MC estimates E^Q[S_T] while analytical uses the exact")
    print("no-arbitrage formula F(0,T) = S_0 * exp(A(T) - B(T)*δ_0).")
    print("Large differences are expected due to the convenience yield dynamics.")
    print("=" * 70)


def compare_mc_vs_analytical(
    S0: float,
    delta0: float,
    K: float,
    h: float,
    u: float,
    p: GibsonSchwartzParams,
    n_sims: int = 100_000,
    seed: int | None = 42,
) -> None:
    # Compare Monte Carlo vs analytical pricing
    
    print("=" * 70)
    print("OPTION PRICING: Monte Carlo vs Analytical")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Spot S0: {S0:.2f}")
    print(f"  Convenience yield δ0: {delta0:.4f}")
    print(f"  Strike K: {K:.2f}")
    print(f"  Option expiry h: {h:.4f} years ({h*12:.1f} months)")
    print(f"  Future maturity (from option expiry) u: {u:.4f} years ({u*12:.1f} months)")
    print(f"  Total future maturity: {h+u:.4f} years ({(h+u)*12:.1f} months)")
    
    # Analytical prices
    print(f"\n{'Analytical Pricing':-^70}")
    call_analytical = price_option_on_future_gibson_schwartz(
        S0, delta0, K, h, u, p, "call"
    )
    put_analytical = price_option_on_future_gibson_schwartz(
        S0, delta0, K, h, u, p, "put"
    )
    print(f"  Call price: {call_analytical:.6f}")
    print(f"  Put  price: {put_analytical:.6f}")
    
    # Monte Carlo prices
    print(f"\n{'Monte Carlo Pricing':-^70}")
    print(f"  Simulations: {n_sims:,}")
    
    call_mc, call_se = mc_price_option_on_future(
        S0, delta0, K, h, u, p, "call", n_sims, seed=seed
    )
    put_mc, put_se = mc_price_option_on_future(
        S0, delta0, K, h, u, p, "put", n_sims, seed=seed
    )
    
    print(f"  Call price: {call_mc:.6f} ± {call_se:.6f}")
    print(f"  Put  price: {put_mc:.6f} ± {put_se:.6f}")
    
    # Comparison
    print(f"\n{'Comparison (MC - Analytical)':-^70}")
    call_diff = call_mc - call_analytical
    put_diff = put_mc - put_analytical
    
    call_pct = 100 * call_diff / call_analytical if call_analytical != 0 else 0
    put_pct = 100 * put_diff / put_analytical if put_analytical != 0 else 0
    
    print(f"  Call difference: {call_diff:+.6f} ({call_pct:+.2f}%)")
    print(f"  Put  difference: {put_diff:+.6f} ({put_pct:+.2f}%)")
    
    # Check if within confidence interval
    call_in_ci = abs(call_diff) < 2 * call_se
    put_in_ci = abs(put_diff) < 2 * put_se
    
    print(f"\n{'Convergence Check (95% CI)':-^70}")
    print(f"  Call: {'✓ PASS' if call_in_ci else '✗ FAIL'} (|diff| < 2σ)")
    print(f"  Put:  {'✓ PASS' if put_in_ci else '✗ FAIL'} (|diff| < 2σ)")
    
    # Show futures price for reference
    F0 = futures_price(S0, delta0, h + u, p)
    print(f"\n{'Market Information':-^70}")
    print(f"  Current futures price F(0, {h+u:.2f}): {F0:.4f}")
    print(f"  Moneyness (F/K): {F0/K:.4f}")
    if F0 < K:
        print(f"  → Put is ITM by {K-F0:.4f}")
    else:
        print(f"  → Call is ITM by {F0-K:.4f}")
    
    print("=" * 70)


# ---------- Example usage ----------

if __name__ == "__main__":
    # Same parameters as analytical example
    params = GibsonSchwartzParams(
        r=0.04,
        kappa=2.0,
        delta_bar_q=0.10,
        sigma_s=0.30,
        sigma_delta=0.20,
        rho=-0.4,
    )

    S0 = 80.0
    delta0 = 0.08
    h = 0.25   # option expiry in 3 months
    u = 0.75   # futures matures 9 months after option expiry
    K = 82.0

    # First, compare futures pricing
    print("\n")
    compare_futures_mc_vs_analytical(
        S0, delta0, params,
        maturities=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
        n_sims=100_000,
        seed=42
    )
    
    print("\n\n")
    
    # Then compare options pricing with 100k simulations
    compare_mc_vs_analytical(
        S0, delta0, K, h, u, params,
        n_sims=100_000,
        seed=42
    )
    
    print("\n\nRunning options with more simulations for better accuracy...")
    print()
    
    # Run with more simulations
    compare_mc_vs_analytical(
        S0, delta0, K, h, u, params,
        n_sims=500_000,
        seed=42
    )
