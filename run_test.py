import pandas as pd
import numpy as np
from dataclasses import replace

# Imports from your local modules
from params import GibsonSchwartzParams
from calibration import get_calibrated_parameters
from gs_model_pricer import (
    price_option_on_future_gibson_schwartz, 
    futures_price
)
from monte_carlo_pricer import monte_carlo_gs_price
from fourier_pricer import price_option_fourier

FILE_NAME = "Data_GS.xlsx"

try:
    # --- 1. LOADING AND CLEANING ---
    df = pd.read_excel(FILE_NAME)
    # Removing rows with missing prices or interest rates
    df = df.dropna(subset=['Close LCOc1', 'Close LCOc2', 'r_1m']).copy()
    
    # --- 2. CALIBRATION ---
    p = get_calibrated_parameters(FILE_NAME)
    
    # --- 3. MARKET STATE (Last available row) ---
    latest = df.iloc[-1]
    S0 = float(latest['Close LCOc1'])
    F2 = float(latest['Close LCOc2'])
    r_actuel = float(latest['r_1m']) # Expected in decimal (e.g., 0.0375)
    
    # --- 4. CALCULATING INITIAL CONVENIENCE YIELD (delta0) ---
    tau1, tau2 = 1/12, 2/12
    delta0 = r_actuel - np.log(F2 / S0) / (tau2 - tau1)

    # --- 5. OPTION PARAMETERS ---
    K = 80.0    # Strike Price
    h = 0.25    # Option maturity (3 months)
    u = 0.5     # Time to maturity of the future at option expiry (6 months)
    
    # Update interest rate in the parameter set
    p_final = replace(p, r=r_actuel)

    # --- 6. PRICING THE UNDERLYING FUTURE ---
    # The future expires at T = h + u
    F0 = futures_price(S0, delta0, h + u, p_final)

    # --- 7. ANALYTICAL CALCULATION (Reference) ---
    call_price = price_option_on_future_gibson_schwartz(S0, delta0, K, h, u, p_final, option="call")

    # --- DISPLAYING FULL RESULTS ---
    print("\n" + "="*60)
    print(f"GIBSON-SCHWARTZ PRICING & CALIBRATION REPORT")
    print("="*60)
    print(f"MARKET DATA DATE    : {latest['Exchange Date']}")
    print("-" * 60)
    print(f"CURRENT MARKET STATE:")
    print(f"  Spot Price (S0)   : {S0:.2f} $")
    print(f"  Conv. Yield (δ0)  : {delta0:.4f}")
    print(f"  Underlying Future : {F0:.4f} $ (T = {h+u:.2f} years)")
    print(f"  Option Strike (K) : {K:.2f} $")
    print("-" * 60)
    print(f"CALIBRATED PARAMETERS:")
    print(f"  Kappa (κ)         : {p_final.kappa:.4f} (Mean Reversion Speed)")
    print(f"  Alpha (α*)        : {p_final.delta_bar_q:.4f} (Long-term Target)")
    print(f"  Sigma S (σs)      : {p_final.sigma_s*100:.2f} % (Spot Volatility)")
    print(f"  Sigma Delta (σδ)  : {p_final.sigma_delta*100:.2f} % (Yield Volatility)")
    print(f"  Rho (ρ)           : {p_final.rho:.4f} (Correlation)")
    print(f"  Risk-free Rate (r): {p_final.r*100:.2f} %")
    print("-" * 60)
    print(f"ANALYTICAL CALL PRICE: {call_price:.4f} $")
    print("="*60)

    # --- 8. MONTE CARLO VALIDATION ---
    print("\n--- MONTE CARLO VALIDATION ---")
    mc_price, mc_std = monte_carlo_gs_price(S0, delta0, K, h, u, p_final, n_sims=100000, n_steps=250)
    print(f"MC Price            : {mc_price:.4f} $ (+/- {mc_std:.4f})")
    diff_mc = abs(call_price - mc_price)
    print(f"MC/Analyt. Diff     : {diff_mc:.6f} $")
    if diff_mc < 0.05:
        print("MC Validation: Passed")
    else:
        print("MC Validation: Discrepancy detected")

    # --- 9. FOURIER VALIDATION ---
    print("\n--- FOURIER VALIDATION ---")
    fourier_price = price_option_fourier(
        S0, delta0, K, h, u, p_final, 
        option="call", 
        xi_max=500.0, 
        n_points=4000
    )
    print(f"Fourier Price       : {fourier_price:.4f} $")
    diff_fourier = abs(call_price - fourier_price)
    print(f"Fou/Analyt. Diff    : {diff_fourier:.6f} $")
    if diff_fourier < 0.001:
        print("Fourier Validation: Passed")
    else:
        print("Fourier Validation: Discrepancy detected")

    print("\n" + "="*60)

except Exception as e:
    print(f"\n CALCULATION ERROR: {e}")
    import traceback
    traceback.print_exc()

# ========================================
# KALMAN FILTER ANALYSIS (TimesSeries_BRENT.xlsx)
# ========================================
print("\n\n" + "="*70)
print("RUNNING KALMAN FILTER ANALYSIS - BRENT CRUDE OIL")
print("="*70)
print("\n NOTE: This section uses TimesSeries_BRENT.xlsx (different dataset)")
print("="*70)

try:
    from Kalman_filter import main
    main()
except Exception as e:
    print(f"\n KALMAN FILTER ERROR: {e}")
    import traceback
    traceback.print_exc()