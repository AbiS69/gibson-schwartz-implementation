"""
Fourier pricer for options on futures in the Gibson-Schwartz model.

Theory recap
------------
Because ln S_T is normally distributed under Q (linear combination of two
Gaussian processes), its characteristic function has a closed-form expression:

    phi(xi) = E^Q[ exp(i * xi * ln F_h) ]
            = exp( i*xi*m - 0.5 * xi^2 * v )

where
    m = E^Q[ln F_h]    (mean of log-futures at option expiry)
    v = Var^Q[ln F_h]  (variance of log-futures at option expiry)

Pricing via Fourier / Gil-Pelaez inversion
------------------------------------------
A European call on futures F with strike K, expiring at h:

    C = e^{-r*h} * [ F0 * Pi1 - K * Pi2 ]

where Pi1 and Pi2 are risk-neutral probabilities recovered from the CF:

    Pi_j = 1/2 + (1/pi) * integral_0^inf Re[ e^{-i*xi*ln(K)} * phi_j(xi) / (i*xi) ] dxi

with
    phi2(xi) = phi(xi)                  (standard CF)
    phi1(xi) = phi(xi - i) / phi(-i)    (measure-shifted CF, i.e. F-measure)

The integral is evaluated by standard numerical quadrature (scipy.integrate or
simple trapezoidal rule on a truncated domain).
"""

from __future__ import annotations

import numpy as np
from math import exp, log, sqrt, pi
from typing import Literal

from gs_model_pricer import (
    GibsonSchwartzParams,
    futures_price,
    var_log_future,
    B,
    price_option_on_future_gibson_schwartz,
    _var_logS,
)


# ---------------------------------------------------------------------------
# 1.  Mean of ln F(h, h+u) under Q
# ---------------------------------------------------------------------------

def mean_log_future(
    S0: float,
    delta0: float,
    h: float,
    u: float,
    p: GibsonSchwartzParams,
) -> float:
    """
    E^Q[ ln F(h, h+u) ] given state (S0, delta0) today.

    ln F(h, h+u) = ln S_h + A(u) - B(u)*delta_h

    Under Q:
        E[ln S_h]   = ln S_0 + (r - delta0 - 0.5*sigma_s^2)*h
                      - (delta0 - delta_bar_q)*(1-e^{-k*h})/k
                      + delta_bar_q * (B(u) from A formula — already in A(h,p))
        Simpler: use E[ln F(h,h+u)] = ln F(0, h+u) - 0.5 * v(h,u)
        because F is a Q-martingale => E^Q[F_h] = F_0,
        but for the LOG we need the Ito correction.

    The cleanest formula:
        ln F(0, h+u) = E^Q[ln F(h,h+u)] + 0.5 * v(h,u)
    so:
        E^Q[ln F(h,h+u)] = ln F(0, h+u) - 0.5 * v(h,u)
    """
    F0 = futures_price(S0, delta0, h + u, p)
    v = var_log_future(h, u, p)
    return log(F0) - 0.5 * v


# ---------------------------------------------------------------------------
# 2.  Characteristic function of ln F(h, h+u)
# ---------------------------------------------------------------------------

def char_func_log_future(
    xi: complex,
    S0: float,
    delta0: float,
    h: float,
    u: float,
    p: GibsonSchwartzParams,
) -> complex:
    """
    Characteristic function of ln F(h, h+u) under Q evaluated at xi (complex).

    Since ln F(h,h+u) ~ N(m, v):
        phi(xi) = exp( i*xi*m - 0.5*xi^2*v )
    """
    m = mean_log_future(S0, delta0, h, u, p)
    v = var_log_future(h, u, p)
    return np.exp(1j * xi * m - 0.5 * xi**2 * v)


# ---------------------------------------------------------------------------
# 3.  Characteristic function of ln S_T and futures pricing via CF
# ---------------------------------------------------------------------------

def mean_log_spot(
    S0: float,
    delta0: float,
    T: float,
    p: GibsonSchwartzParams,
) -> float:
    """
    E^Q[ ln S_T ] given state (S0, delta0) today.

    Since F(0,T) = E^Q[S_T] * e^{r*T}  (no — this is wrong for GBM with
    convenience yield; the correct relation uses the Ito correction):

        E^Q[ln S_T] = ln F(0,T) - 0.5 * Var^Q[ln S_T]

    which follows from ln S_T ~ N(m_S, v_S) and F(0,T) = exp(m_S + 0.5*v_S).
    """
    F0 = futures_price(S0, delta0, T, p)
    v  = _var_logS(T, p)
    return log(F0) - 0.5 * v


def char_func_log_spot(
    xi: complex,
    S0: float,
    delta0: float,
    T: float,
    p: GibsonSchwartzParams,
) -> complex:
    """
    Characteristic function of ln S_T under Q at xi (complex).

    Since ln S_T ~ N(m_S, v_S):
        phi(xi) = exp( i*xi*m_S - 0.5*xi^2*v_S )

    Key identity used for futures pricing:
        E^Q[S_T] = E^Q[e^{ln S_T}] = phi(-i)
    """
    m = mean_log_spot(S0, delta0, T, p)
    v = _var_logS(T, p)
    return np.exp(1j * xi * m - 0.5 * xi**2 * v)


def futures_price_cf(
    S0: float,
    delta0: float,
    T: float,
    p: GibsonSchwartzParams,
) -> float:
    """
    Futures price F(0,T) via the characteristic function.

    F(0,T) = e^{r*T} * E^Q[S_T]
           = e^{r*T} * phi_logS(-i)

    where phi_logS(-i) = E^Q[e^{ln S_T}] = E^Q[S_T].
    """
    E_S_T = float(np.real(char_func_log_spot(-1j, S0, delta0, T, p)))
    return exp(p.r * T) * E_S_T


def compare_futures_cf_vs_analytical(
    S0: float,
    delta0: float,
    p: GibsonSchwartzParams,
    maturities: list[float] | None = None,
) -> None:
    """Compare futures prices: CF method vs closed-form analytical."""
    if maturities is None:
        maturities = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

    print("=" * 70)
    print("FUTURES PRICING: Characteristic Function vs Analytical")
    print("=" * 70)
    print(f"\n  Spot S0 = {S0:.2f}  |  δ0 = {delta0:.4f}\n")
    print(f"  {'Maturity':>10s} {'Analytical':>15s} {'CF method':>15s} {'Diff':>12s} {'Rel%':>8s}")
    print("  " + "-" * 64)

    for T in maturities:
        F_an = futures_price(S0, delta0, T, p)
        F_cf = futures_price_cf(S0, delta0, T, p)
        diff = F_cf - F_an
        rel  = abs(diff) / F_an * 100
        print(f"  {T:>10.2f}y {F_an:>15.6f} {F_cf:>15.6f} {diff:>+12.6f} {rel:>7.4f}%")

    print("=" * 70)


# ---------------------------------------------------------------------------
# 4.  Fourier inversion via numerical quadrature (Gil-Pelaez)
# ---------------------------------------------------------------------------

def _integrand_pi(
    xi: float,
    k: float,
    j: int,
    S0: float,
    delta0: float,
    h: float,
    u: float,
    p: GibsonSchwartzParams,
) -> float:
    """
    Integrand for Pi_j:
        Re[ e^{-i*xi*k} * phi_j(xi) / (i*xi) ]

    j=2 : phi2(xi) = phi(xi)
    j=1 : phi1(xi) = phi(xi - i) / phi(-i)   [F-measure shift]
    """
    if j == 2:
        cf = char_func_log_future(xi + 0j, S0, delta0, h, u, p)
    else:  # j == 1
        cf_shift    = char_func_log_future(xi - 1j, S0, delta0, h, u, p)
        cf_normalise = char_func_log_future(-1j,    S0, delta0, h, u, p)
        cf = cf_shift / cf_normalise

    # e^{-i*xi*k} * cf / (i*xi)
    val = np.exp(-1j * xi * k) * cf / (1j * xi)
    return float(np.real(val))


def _pi_j(
    j: int,
    K: float,
    S0: float,
    delta0: float,
    h: float,
    u: float,
    p: GibsonSchwartzParams,
    xi_max: float = 200.0,
    n_points: int = 2000,
) -> float:
    """
    Pi_j = 1/2 + (1/pi) * integral_0^{xi_max} integrand dxi

    Evaluated by trapezoidal rule on n_points uniform points.
    """
    k = log(K)
    xi_grid = np.linspace(1e-8, xi_max, n_points)
    dxi     = xi_grid[1] - xi_grid[0]

    vals = np.array([_integrand_pi(xi, k, j, S0, delta0, h, u, p) for xi in xi_grid])
    integral = np.trapezoid(vals, xi_grid)
    return 0.5 + integral / pi


def price_option_fourier(
    S0: float,
    delta0: float,
    K: float,
    h: float,
    u: float,
    p: GibsonSchwartzParams,
    option: Literal["call", "put"] = "call",
    xi_max: float = 200.0,
    n_points: int = 2000,
) -> float:
    """
    Price a European option on futures at strike K using Fourier inversion.

    C = e^{-r*h} * [ F0 * Pi1 - K * Pi2 ]
    P = C - e^{-r*h} * (F0 - K)          (put-call parity)
    """
    F0 = futures_price(S0, delta0, h + u, p)
    df = exp(-p.r * h)

    pi1 = _pi_j(1, K, S0, delta0, h, u, p, xi_max, n_points)
    pi2 = _pi_j(2, K, S0, delta0, h, u, p, xi_max, n_points)

    call = df * (F0 * pi1 - K * pi2)
    call = max(call, 0.0)

    if option == "call":
        return call
    else:
        put = call - df * (F0 - K)
        return max(put, 0.0)


# ---------------------------------------------------------------------------
# 5.  Comparison helper (options)
# ---------------------------------------------------------------------------

def compare_fourier_vs_analytical(
    S0: float,
    delta0: float,
    K: float,
    h: float,
    u: float,
    p: GibsonSchwartzParams,
    xi_max: float = 200.0,
    n_points: int = 2000,
) -> None:
    """Print a full comparison table: Fourier vs Analytical."""

    print("=" * 70)
    print("OPTION PRICING: Fourier (Gil-Pelaez) vs Analytical (Black-76 / GS)")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Spot S0          : {S0:.2f}")
    print(f"  Convenience yield: {delta0:.4f}")
    print(f"  Strike K         : {K:.2f}")
    print(f"  Option expiry h  : {h:.4f} yr  ({h*12:.1f} months)")
    print(f"  Future horizon u : {u:.4f} yr  ({u*12:.1f} months)")
    print(f"  Total maturity   : {h+u:.4f} yr  ({(h+u)*12:.1f} months)")
    print(f"  Integration pts  : {n_points:,}  (xi in [0, {xi_max}])")

    F0 = futures_price(S0, delta0, h + u, p)
    m  = mean_log_future(S0, delta0, h, u, p)
    v  = var_log_future(h, u, p)

    print(f"\n{'Derived quantities':-^70}")
    print(f"  F(0, h+u)        : {F0:.6f}")
    print(f"  E[ln F_h]  m(h)  : {m:.6f}")
    print(f"  Var[ln F_h] v(h) : {v:.6f}")
    print(f"  Implied vol σ    : {sqrt(v/h)*100:.4f}%  (annualised sqrt(v/h))")

    # Analytical
    call_an = price_option_on_future_gibson_schwartz(S0, delta0, K, h, u, p, "call")
    put_an  = price_option_on_future_gibson_schwartz(S0, delta0, K, h, u, p, "put")

    # Fourier
    call_fou = price_option_fourier(S0, delta0, K, h, u, p, "call", xi_max, n_points)
    put_fou  = price_option_fourier(S0, delta0, K, h, u, p, "put",  xi_max, n_points)

    print(f"\n{'Results':^70}")
    print(f"{'':20s} {'Call':>15s} {'Put':>15s}")
    print(f"  {'Analytical':<18s} {call_an:>15.6f} {put_an:>15.6f}")
    print(f"  {'Fourier':<18s} {call_fou:>15.6f} {put_fou:>15.6f}")
    print(f"  {'Difference':<18s} {call_fou-call_an:>+15.6f} {put_fou-put_an:>+15.6f}")

    call_rel = abs(call_fou - call_an) / call_an * 100 if call_an else 0
    put_rel  = abs(put_fou  - put_an)  / put_an  * 100 if put_an  else 0
    print(f"  {'Rel. error (%)':<18s} {call_rel:>14.4f}% {put_rel:>14.4f}%")

    print("=" * 70)


def compare_fourier_strikes(
    S0: float,
    delta0: float,
    h: float,
    u: float,
    p: GibsonSchwartzParams,
    strikes: list[float] | None = None,
    xi_max: float = 200.0,
    n_points: int = 2000,
) -> None:
    """Compare Fourier vs Analytical across a range of strikes."""

    F0 = futures_price(S0, delta0, h + u, p)

    if strikes is None:
        strikes = [round(F0 * m, 4) for m in
                   [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]]

    print("=" * 80)
    print("CALL PRICES ACROSS STRIKES: Fourier vs Analytical")
    print(f"  F0 = {F0:.4f}  |  h = {h:.2f}y  |  u = {u:.2f}y")
    print("=" * 80)
    print(f"{'Strike':>10s} {'Moneyness':>10s} {'Analytical':>14s} "
          f"{'Fourier':>14s} {'Diff':>12s} {'Rel%':>8s}")
    print("-" * 80)

    for K in strikes:
        call_an  = price_option_on_future_gibson_schwartz(S0, delta0, K, h, u, p, "call")
        call_fou = price_option_fourier(S0, delta0, K, h, u, p, "call", xi_max, n_points)
        diff = call_fou - call_an
        rel  = abs(diff) / call_an * 100 if call_an > 1e-10 else 0.0
        print(f"{K:>10.4f} {K/F0:>10.4f} {call_an:>14.6f} "
              f"{call_fou:>14.6f} {diff:>+12.6f} {rel:>7.4f}%")

    print("=" * 80)


# ---------------------------------------------------------------------------
# 6.  Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    params = GibsonSchwartzParams(
        r=0.04,
        kappa=2.0,
        delta_bar_q=0.10,
        sigma_s=0.30,
        sigma_delta=0.20,
        rho=-0.4,
    )

    S0     = 80.0
    delta0 = 0.08
    h      = 0.25    # option expiry: 3 months
    u      = 0.75    # futures maturity 9 months after option expiry
    K      = 82.0

    # 1) Futures pricing via CF
    compare_futures_cf_vs_analytical(S0, delta0, params)

    print("\n")

    # 2) Single-strike option comparison
    compare_fourier_vs_analytical(S0, delta0, K, h, u, params)

    print("\n")

    # 3) Smile across strikes
    compare_fourier_strikes(S0, delta0, h, u, params)

