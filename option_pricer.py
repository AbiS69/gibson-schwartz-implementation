from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt, erf, isfinite
from typing import Literal


def _norm_cdf(x: float) -> float:
    # Standard normal CDF using erf (no SciPy dependency).
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


@dataclass(frozen=True)
class GibsonSchwartzParams:
    # Gibson–Schwartz / Schwartz(1997) 2-factor (Gaussian) model parameters under Q.
    #
    # dS/S = (r - delta) dt + sigma_s dW1
    # ddelta = kappa (delta_bar_q - delta) dt + sigma_delta dW2
    # corr(dW1, dW2) = rho dt
    r: float                 # risk-free rate (constant)
    kappa: float             # mean reversion speed of convenience yield
    delta_bar_q: float       # long-run mean of delta under Q
    sigma_s: float           # spot volatility (constant)
    sigma_delta: float       # delta volatility (constant)
    rho: float               # instantaneous correlation in [-1, 1]


# ---------- Futures pricing: F(t,T) = S_t * exp(A(tau) - B(tau)*delta_t) ----------

def B(tau: float, kappa: float) -> float:
    # Affine loading B(tau) for OU convenience yield.
    if tau < 0:
        raise ValueError("tau must be >= 0")
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    return (1.0 - exp(-kappa * tau)) / kappa


def _V_integral_delta(tau: float, p: GibsonSchwartzParams) -> float:
    # Variance correction term appearing in A(tau), coming from:
    # ln F(t,T) = ln S_t + A(tau) - B(tau) delta_t
    k = p.kappa
    sd = p.sigma_delta
    ss = p.sigma_s
    rho = p.rho

    e1 = exp(-k * tau)
    e2 = exp(-2.0 * k * tau)

    term_delta = sd * sd * (
        (tau / (k * k))
        - (2.0 * (1.0 - e1) / (k * k * k))
        + ((1.0 - e2) / (2.0 * k * k * k))
    )

    term_cross = -2.0 * rho * ss * sd * (
        (tau / k)
        - ((1.0 - e1) / (k * k))
    )

    return term_delta + term_cross


def A(tau: float, p: GibsonSchwartzParams) -> float:
    # Affine term A(tau) for futures price under Q.
    if tau < 0:
        raise ValueError("tau must be >= 0")
    b = B(tau, p.kappa)
    v = _V_integral_delta(tau, p)
    return p.r * tau - p.delta_bar_q * (tau - b) + 0.5 * v


def futures_price(S_t: float, delta_t: float, tau: float, p: GibsonSchwartzParams) -> float:
    # F(t,T) with tau = T - t.
    if S_t <= 0:
        raise ValueError("S_t must be > 0")
    if tau < 0:
        raise ValueError("tau must be >= 0")
    b = B(tau, p.kappa)
    a = A(tau, p)
    return S_t * exp(a - b * delta_t)


# ---------- Variance of log-future over option horizon ----------

def _var_delta(h: float, p: GibsonSchwartzParams) -> float:
    # Var_t(delta_{t+h}) for OU.
    k = p.kappa
    sd = p.sigma_delta
    return sd * sd * (1.0 - exp(-2.0 * k * h)) / (2.0 * k)


def _var_logS(h: float, p: GibsonSchwartzParams) -> float:
    # Var_t(ln S_{t+h}) under Q when d ln S includes -delta dt and delta is OU.
    k = p.kappa
    ss = p.sigma_s
    sd = p.sigma_delta
    rho = p.rho

    e1 = exp(-k * h)
    e2 = exp(-2.0 * k * h)

    A1 = h - 2.0 * (1.0 - e1) / k + (1.0 - e2) / (2.0 * k)
    A2 = h - (1.0 - e1) / k

    return (ss * ss) * h + (sd * sd / (k * k)) * A1 - 2.0 * (ss * sd * rho / k) * A2


def _cov_logS_delta(h: float, p: GibsonSchwartzParams) -> float:
    # Cov_t(ln S_{t+h}, delta_{t+h}) under Q.
    k = p.kappa
    ss = p.sigma_s
    sd = p.sigma_delta
    rho = p.rho

    e1 = exp(-k * h)
    one_minus = 1.0 - e1

    return (ss * sd * rho / k) * one_minus - (sd * sd / (2.0 * k * k)) * (one_minus ** 2)


def var_log_future(h: float, u: float, p: GibsonSchwartzParams) -> float:
    # v = Var_t( ln F(t+h, t+h+u) )
    # where h = option time-to-expiry, u = (future maturity - option expiry).
    if h < 0 or u < 0:
        raise ValueError("h and u must be >= 0")

    b = B(u, p.kappa)  # loading at option expiry on delta_{t+h}

    v_logS = _var_logS(h, p)
    v_delta = _var_delta(h, p)
    cov = _cov_logS_delta(h, p)

    v = v_logS + (b * b) * v_delta - 2.0 * b * cov
    return max(0.0, v)  # guard against tiny negative due to rounding


# ---------- Black-76 option on futures using integrated log-variance v ----------

def black76(
    F0: float,
    K: float,
    r: float,
    h: float,
    v: float,
    option: Literal["call", "put"] = "call",
) -> float:
    # Black-76 with log-variance v = Var(ln F_T) over horizon h.
    # Discounting uses exp(-r h).
    if F0 <= 0 or K <= 0:
        raise ValueError("F0 and K must be > 0")
    if h < 0:
        raise ValueError("h must be >= 0")
    if v < 0:
        raise ValueError("v must be >= 0")

    df = exp(-r * h)

    if v == 0.0 or h == 0.0:
        intrinsic = max(F0 - K, 0.0) if option == "call" else max(K - F0, 0.0)
        return df * intrinsic

    s = sqrt(v)
    d1 = (log(F0 / K) + 0.5 * v) / s
    d2 = d1 - s

    if option == "call":
        return df * (F0 * _norm_cdf(d1) - K * _norm_cdf(d2))
    elif option == "put":
        return df * (K * _norm_cdf(-d2) - F0 * _norm_cdf(-d1))
    else:
        raise ValueError("option must be 'call' or 'put'")


# ---------- End-to-end wrapper: option on futures in Gibson–Schwartz ----------

def price_option_on_future_gibson_schwartz(
    S_t: float,
    delta_t: float,
    K: float,
    h: float,
    u: float,
    p: GibsonSchwartzParams,
    option: Literal["call", "put"] = "call",
) -> float:
    # Prices a European option expiring in h years on a futures contract with
    # remaining maturity u years at option expiry.
    #
    # - Today: t
    # - Option expiry: tau = t + h
    # - Futures maturity: T = tau + u
    #
    # Inputs:
    #   S_t, delta_t : today's state
    #   K            : strike on the futures price
    #   h            : option time-to-expiry (years)
    #   u            : (future maturity - option expiry) (years)
    
    # 1) Current futures level for maturity (h+u)
    F0 = futures_price(S_t, delta_t, tau=h + u, p=p)

    # 2) Risk-neutral log-variance of the futures price at option expiry
    v = var_log_future(h=h, u=u, p=p)

    # 3) Black-76
    return black76(F0=F0, K=K, r=p.r, h=h, v=v, option=option)


# ---------- Example usage ----------

if __name__ == "__main__":
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
    u = 0.75   # futures matures 9 months after option expiry (so total 1 year from now)
    K = 82.0

    call_px = price_option_on_future_gibson_schwartz(S0, delta0, K, h, u, params, "call")
    put_px  = price_option_on_future_gibson_schwartz(S0, delta0, K, h, u, params, "put")

    print(f"Call price: {call_px:.6f}")
    print(f"Put  price: {put_px:.6f}")
