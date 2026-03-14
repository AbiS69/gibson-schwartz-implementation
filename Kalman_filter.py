"""
Gibson-Schwartz (1997) Two-Factor Commodity Model
Kalman Filter with Maximum Likelihood Estimation

This implementation follows Schwartz (1997) for analytical formulas of A(tau) and B(tau).
Numerically robust version using pseudo-inverse and parameter bounds.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')


class GibsonSchwartz:
    """
    Gibson-Schwartz two-factor commodity model with Kalman Filter and MLE.
    """
    
    def __init__(self, spot_prices, futures_prices, maturities, risk_free_rates, dt=1/252):
        """
        Initialize the model.
        
        Parameters:
        -----------
        spot_prices : array-like
            Spot prices (S_t)
        futures_prices : array-like of shape (n_obs, n_contracts)
            Futures prices for different maturities
        maturities : array-like
            Time to maturity in years for each futures contract
        risk_free_rates : array-like
            Risk-free rates (r_t) as decimal, synchronized with spot prices
        dt : float
            Time step (default: 1/252 for daily data)
        """
        self.spot_prices = np.asarray(spot_prices).astype(float)
        self.futures_prices = np.asarray(futures_prices).astype(float)
        self.maturities = np.asarray(maturities).astype(float)
        self.risk_free_rates = np.asarray(risk_free_rates).astype(float)
        self.dt = dt
        self.n_obs = len(spot_prices)
        self.n_contracts = len(maturities) if futures_prices.ndim > 1 else 1
        
        # Ensure risk_free_rates are synchronized with observations
        if len(self.risk_free_rates) != self.n_obs:
            raise ValueError(f"risk_free_rates length ({len(self.risk_free_rates)}) must match observations ({self.n_obs})")
        
        # Convert to log-spot and log-futures
        self.log_spot = np.log(self.spot_prices)
        if futures_prices.ndim == 1:
            self.log_futures = np.log(futures_prices)[:, None]
        else:
            self.log_futures = np.log(futures_prices)
        
        self.params = None
        self.filtered_states = None
        
    def B_tau(self, tau, kappa):
        """Compute B(tau) from Schwartz (1997)."""
        if kappa < 1e-6:
            return tau
        return (1.0 - np.exp(-kappa * tau)) / kappa
    
    def A_tau(self, tau, kappa, sigma_1, sigma_2, rho, r, delta_bar_q):
        """Compute A(tau) from Schwartz (1997)."""
        k = kappa
        ss = sigma_1
        sd = sigma_2
        
        e1 = np.exp(-k * tau)
        e2 = np.exp(-2.0 * k * tau)
        
        b = self.B_tau(tau, k)
        
        # Variance correction term
        term_delta = sd**2 * (
            (tau / (k**2))
            - (2.0 * (1.0 - e1) / (k**3))
            + ((1.0 - e2) / (2.0 * k**3))
        )
        
        term_cross = -2.0 * rho * ss * sd * (
            (tau / k) - ((1.0 - e1) / (k**2))
        )
        
        v = term_delta + term_cross
        
        return r * tau - delta_bar_q * (tau - b) + 0.5 * v
    
    def kalman_filter(self, params):
        """
        Run Kalman filter for likelihood computation.
        Uses numerically stable operations.
        
        Parameters:
        -----------
        params : array
            [kappa, sigma_1, sigma_2, rho, lambda, mu_s, delta_bar_q]
            r_t is obtained from self.risk_free_rates
        """
        # Unpack and clip parameters
        kappa = np.clip(params[0], 0.01, 5.0)
        sigma_1 = np.clip(params[1], 0.01, 2.0)
        sigma_2 = np.clip(params[2], 0.01, 2.0)
        rho = np.clip(params[3], -0.99, 0.99)
        lambda_param = np.clip(params[4], -0.5, 0.5)
        mu_s = np.clip(params[5], 0.0, 1.0)
        delta_bar_q = np.clip(params[6], 0.0, 1.0)
        
        try:
            # Build transition matrices (G and c for discrete-time)
            exp_k = np.exp(-kappa * self.dt)
            G = np.array([
                [1.0, -self.dt],
                [0.0, exp_k]
            ])
            
            # Process noise covariance Q (independent of time-varying r)
            var_x = sigma_1**2 * self.dt
            var_delta = sigma_2**2 * (1.0 - np.exp(-2.0 * kappa * self.dt)) / (2.0 * kappa + 1e-8)
            cov_x_delta = rho * sigma_1 * sigma_2 * (1.0 - exp_k) / (kappa + 1e-8)
            
            Q = np.array([
                [var_x, cov_x_delta],
                [cov_x_delta, var_delta]
            ])
            
            # Ensure Q is positive definite
            Q = (Q + Q.T) / 2
            eigs = np.linalg.eigvalsh(Q)
            if np.any(eigs < 1e-8):
                Q += np.eye(2) * (1e-7 - np.min(eigs))
            
            # Build measurement matrices
            Z = np.zeros((self.n_contracts, 2))
            Z[:, 0] = 1.0
            Z[:, 1] = -self.B_tau(self.maturities, kappa)
            
            # Measurement noise covariance (larger for stability)
            H = np.eye(self.n_contracts) * 1e-3
            
            # Initialize filter
            x = np.array([self.log_spot[0], mu_s])
            P = np.eye(2) * 100.0
            
            log_likelihood = 0.0
            n_valid = 0
            
            # Kalman filter loop
            for t in range(self.n_obs):
                # Get time-varying risk-free rate
                r_t = self.risk_free_rates[t]
                
                # Update drift term in state equation with r_t and compute d with r_t
                c = np.array([
                    (r_t - lambda_param - 0.5 * sigma_1**2) * self.dt,
                    mu_s * (1.0 - exp_k)
                ])
                
                # Recompute d with current r_t (risk-neutral pricing)
                d = np.array([
                    self.A_tau(tau, kappa, sigma_1, sigma_2, rho, r_t, delta_bar_q)
                    for tau in self.maturities
                ])
                
                # Prediction
                x_pred = c + G @ x
                P_pred = G @ P @ G.T + Q
                
                # Ensure P_pred is symmetric and positive definite
                P_pred = (P_pred + P_pred.T) / 2
                eigs_p = np.linalg.eigvalsh(P_pred)
                if np.any(eigs_p < 1e-10):
                    P_pred += np.eye(2) * (1e-9 - np.min(eigs_p))
                
                # Measurement update
                y_pred = d + Z @ x_pred
                innovation = self.log_futures[t] - y_pred
                
                # Innovation covariance
                S = Z @ P_pred @ Z.T + H
                S = (S + S.T) / 2
                
                # Use pseudo-inverse for numerical stability
                try:
                    S_inv = np.linalg.pinv(S, rcond=1e-10)
                    det_S = np.linalg.det(S)
                    
                    if det_S > 1e-30 and np.isfinite(det_S):
                        # Kalman gain
                        K = P_pred @ Z.T @ S_inv
                        
                        # Update state and covariance
                        x = x_pred + K @ innovation
                        P = (np.eye(2) - K @ Z) @ P_pred
                        
                        # Log-likelihood contribution (more numerically stable)
                        mahal = float(innovation @ S_inv @ innovation)
                        if np.isfinite(mahal) and mahal < 1000:
                            ll_t = -0.5 * (self.n_contracts * np.log(2 * np.pi) + 
                                          np.log(det_S) + mahal)
                            if np.isfinite(ll_t):
                                log_likelihood += ll_t
                                n_valid += 1
                except:
                    pass
            
            if n_valid < 50:  # Need sufficient valid observations
                return 1e10
            
            # Normalize by number of valid observations
            avg_ll = log_likelihood / n_valid
            
            # Return negative log-likelihood for minimization
            return -avg_ll
            
        except Exception as e:
            return 1e10
    
    def fit(self, verbose=True, method='L-BFGS-B'):
        """
        Fit the model using MLE.
        
        Parameters:
        -----------
        verbose : bool
            Print progress
        method : str
            Optimization method
        """
        # Initial parameter guess
        # Parameters: [kappa, sigma_1, sigma_2, rho, lambda, mu_s, delta_bar_q]
        # r_t is now obtained from risk_free_rates vector
        initial_params = np.array([
            0.5,     # kappa
            0.25,    # sigma_1
            0.25,    # sigma_2
            0.3,     # rho
            0.0,     # lambda
            0.05,    # mu_s
            0.05     # delta_bar_q
        ])
        
        # Parameter bounds
        bounds = [
            (0.01, 5.0),      # kappa
            (0.01, 2.0),      # sigma_1
            (0.01, 2.0),      # sigma_2
            (-0.99, 0.99),    # rho
            (-0.5, 0.5),      # lambda
            (0.0, 1.0),       # mu_s
            (0.0, 1.0)        # delta_bar_q
        ]
        
        if verbose:
            print("\nStarting MLE optimization...")
        
        # Optimize
        result = minimize(
            self.kalman_filter,
            initial_params,
            method=method,
            bounds=bounds,
            options={'maxiter': 200, 'disp': False}
        )
        
        if verbose:
            print(f"Optimization completed.")
            print(f"Final log-likelihood: {-result.fun:.6f}")
        
        self.params = result.x
        return result
    
    def print_parameters(self):
        """Print fitted parameters."""
        if self.params is None:
            print("Model not fitted yet.")
            return
        
        param_names = ['kappa', 'sigma_1', 'sigma_2', 'rho', 'lambda', 'mu_s', 'delta_bar_q']
        print("\n" + "="*65)
        print("GIBSON-SCHWARTZ MODEL - FITTED PARAMETERS (BRENT CRUDE OIL)")
        print("="*65)
        for name, value in zip(param_names, self.params):
            print(f"{name:20s}: {value:12.8f}")
        print("="*65)
        
        # Interpret parameters
        print("\nParameter Interpretation:")
        print(f"  κ (mean reversion): {self.params[0]:.4f} - Controls speed of convenience yield reversion")
        print(f"  σ₁ (spot vol):      {self.params[1]:.4f} - Volatility of log-spot price")
        print(f"  σ₂ (delta vol):     {self.params[2]:.4f} - Volatility of convenience yield")
        print(f"  ρ (correlation):    {self.params[3]:.4f} - Correlation between spot and delta shocks")
        print(f"  λ (market price):   {self.params[4]:.4f} - Market price of risk")
        print(f"  μₛ (long-run δ):    {self.params[5]:.4f} - Long-run convenience yield level")
    
    def extract_states(self):
        """
        Extract filtered states using fitted parameters.
        """
        if self.params is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        kappa, sigma_1, sigma_2, rho, lambda_param, mu_s, delta_bar_q = self.params
        
        # Build matrices
        exp_k = np.exp(-kappa * self.dt)
        G = np.array([
            [1.0, -self.dt],
            [0.0, exp_k]
        ])
        
        var_x = sigma_1**2 * self.dt
        var_delta = sigma_2**2 * (1.0 - np.exp(-2.0 * kappa * self.dt)) / (2.0 * kappa + 1e-8)
        cov_x_delta = rho * sigma_1 * sigma_2 * (1.0 - exp_k) / (kappa + 1e-8)
        Q = np.array([[var_x, cov_x_delta], [cov_x_delta, var_delta]])
        Q = (Q + Q.T) / 2
        
        Z = np.zeros((self.n_contracts, 2))
        Z[:, 0] = 1.0
        Z[:, 1] = -self.B_tau(self.maturities, kappa)
        
        H = np.eye(self.n_contracts) * 1e-5
        
        # Initialize
        x = np.array([self.log_spot[0], mu_s])
        P = np.eye(2) * 100.0
        
        states = np.zeros((self.n_obs, 2))
        
        # Filter loop
        for t in range(self.n_obs):
            # Get time-varying risk-free rate
            r_t = self.risk_free_rates[t]
            
            # Update drift term with r_t
            c = np.array([
                (r_t - lambda_param - 0.5 * sigma_1**2) * self.dt,
                mu_s * (1.0 - exp_k)
            ])
            
            # Recompute d with current r_t
            d = np.array([
                self.A_tau(tau, kappa, sigma_1, sigma_2, rho, r_t, delta_bar_q)
                for tau in self.maturities
            ])
            
            x_pred = c + G @ x
            P_pred = G @ P @ G.T + Q
            P_pred = (P_pred + P_pred.T) / 2
            
            y_pred = d + Z @ x_pred
            innovation = self.log_futures[t] - y_pred
            S = Z @ P_pred @ Z.T + H
            S = (S + S.T) / 2
            
            try:
                S_inv = np.linalg.pinv(S)
                K = P_pred @ Z.T @ S_inv
                x = x_pred + K @ innovation
                P = (np.eye(2) - K @ Z) @ P_pred
            except:
                pass
            
            states[t] = x
        
        self.filtered_states = states
        spot_estimated = np.exp(states[:, 0])
        convenience_yield = states[:, 1]
        
        return spot_estimated, convenience_yield
    
    def plot_results(self):
        """Plot spot price and convenience yield."""
        if self.filtered_states is None:
            _, _ = self.extract_states()
        
        spot_est = np.exp(self.filtered_states[:, 0])
        dy = self.filtered_states[:, 1]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Spot Price
        axes[0].plot(self.spot_prices, 'b-', linewidth=2, label='Actual Spot Price', alpha=0.8)
        axes[0].plot(spot_est, 'r--', linewidth=1.5, label='Estimated Spot Price', alpha=0.8)
        axes[0].set_xlabel('Time (days)', fontsize=11)
        axes[0].set_ylabel('Price ($/barrel)', fontsize=11)
        axes[0].set_title('Brent Crude Oil: Actual vs Filtered Spot Price', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10, loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Add statistics
        rmse_spot = np.sqrt(np.mean((self.spot_prices - spot_est)**2))
        axes[0].text(0.98, 0.05, f'RMSE: ${rmse_spot:.4f}/barrel', transform=axes[0].transAxes,
                    fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot 2: Convenience Yield
        axes[1].plot(dy, 'g-', linewidth=1.5, label='Estimated Convenience Yield', alpha=0.8)
        axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        axes[1].fill_between(range(len(dy)), dy, 0, where=(dy >= 0), alpha=0.3, color='green', label='Positive')
        axes[1].fill_between(range(len(dy)), dy, 0, where=(dy < 0), alpha=0.3, color='red', label='Negative')
        axes[1].set_xlabel('Time (days)', fontsize=11)
        axes[1].set_ylabel('Convenience Yield', fontsize=11)
        axes[1].set_title('Brent Crude Oil: Estimated Instantaneous Convenience Yield', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10, loc='best')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gs_model_results.png', dpi=150, bbox_inches='tight')
        print(" Plot saved: gs_model_results.png")
        plt.close()
    
    def plot_futures(self):
        """Plot fitted vs actual futures prices."""
        if self.params is None:
            raise ValueError("Model not fitted.")
        
        kappa, sigma_1, sigma_2, rho, lambda_param, mu_s, delta_bar_q = self.params
        
        Z = np.zeros((self.n_contracts, 2))
        Z[:, 0] = 1.0
        Z[:, 1] = -self.B_tau(self.maturities, kappa)
        
        if self.filtered_states is None:
            _, _ = self.extract_states()
        
        # Compute fitted log-futures for each time step with time-varying r
        fitted_list = []
        for t in range(self.n_obs):
            r_t = self.risk_free_rates[t]
            d = np.array([
                self.A_tau(tau, kappa, sigma_1, sigma_2, rho, r_t, delta_bar_q)
                for tau in self.maturities
            ])
            log_fitted_t = d + Z @ self.filtered_states[t]
            fitted_list.append(np.exp(log_fitted_t))
        
        fitted = np.array(fitted_list)
        
        n_cont = self.n_contracts
        n_rows = (n_cont + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_cont > 1 else [axes]
        
        for i in range(n_cont):
            ax = axes[i]
            actual = self.futures_prices[:, i] if self.futures_prices.ndim > 1 else self.futures_prices
            ax.plot(actual, 'b-', linewidth=2, label='Actual', alpha=0.7)
            ax.plot(fitted[:, i], 'r--', linewidth=1.5, label='Model-fitted', alpha=0.7)
            ax.set_xlabel('Time (days)', fontsize=10)
            ax.set_ylabel('Futures Price ($/barrel)', fontsize=10)
            ax.set_title(f'Brent Contract {i+1} (τ = {self.maturities[i]:.3f} yrs)', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # RMSE
            rmse = np.sqrt(np.mean((actual - fitted[:, i])**2))
            ax.text(0.98, 0.05, f'RMSE: {rmse:.4f}', transform=ax.transAxes,
                   fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        for j in range(n_cont, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('gs_futures_comparison.png', dpi=150, bbox_inches='tight')
        print("Plot saved: gs_futures_comparison.png")
        plt.close()


def main():
    """Main function."""
    
    print("\n" + "="*65)
    print("GIBSON-SCHWARTZ TWO-FACTOR COMMODITY PRICING MODEL")
    print("BRENT CRUDE OIL ANALYSIS")
    print("="*65)
    
    # Load data
    print("\nLoading Brent time series data...")
    try:
        # Try to read from Excel file with 'Feuil1' sheet (French for Sheet1)
        df = pd.read_excel('TimesSeries_BRENT.xlsx', sheet_name='Feuil1')
    except:
        try:
            # Fallback to first sheet if 'Feuil1' doesn't exist
            df = pd.read_excel('TimesSeries_BRENT.xlsx', sheet_name=0)
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return
    
    print(f"  Data loaded successfully")
    print(f"  Available columns: {df.columns.tolist()}")
    
    # Extract spot prices from F1 (first futures contract as proxy for spot)
    if 'F1' not in df.columns:
        print("Error: Column 'F1' not found in data")
        return
    
    # Get clean indices (remove NaN from spot AND risk-free rates simultaneously)
    spot_raw = pd.to_numeric(df['F1'], errors='coerce').values
    
    # Find valid indices (not NaN in spot price)
    valid_idx = ~np.isnan(spot_raw)
    spot = spot_raw[valid_idx].copy()
    
    # Get risk-free rates and apply same valid indices
    risk_free_rates_raw = pd.to_numeric(df['T-Bill_1M'], errors='coerce').values if 'T-Bill_1M' in df.columns else None
    
    if risk_free_rates_raw is None or risk_free_rates_raw.shape[0] < spot_raw.shape[0]:
        print("Warning: Column 'T-Bill_1M' not found or insufficient data. Using default 2% rate.")
        risk_free_rates = np.full(len(spot), 0.02)
    else:
        risk_free_rates = risk_free_rates_raw[valid_idx].copy()
        
        # Convert to decimal if in percentage (assume > 0.5 means percentage)
        if np.nanmax(risk_free_rates) > 0.5:
            risk_free_rates = risk_free_rates / 100.0
            print(" Risk-free rates converted from percentage to decimal")
        
        # Handle any NaN values in risk-free rates by forward-filling
        for i in range(len(risk_free_rates)):
            if np.isnan(risk_free_rates[i]):
                risk_free_rates[i] = risk_free_rates[i-1] if i > 0 else 0.02
    
    # Extract futures prices and maturities
    futures_list = []
    maturities = []
    
    # Futures contracts: F1, F2, F3, F4, F5 with corresponding maturities
    futures_cols = ['F1', 'F2', 'F3', 'F4', 'F5']
    
    # Maturity mapping for Brent crude futures (typical tenor structure)
    maturity_dict = {
        'F1': 1/12,      # 1 month
        'F2': 5/12,      # 5 months
        'F3': 9/12,      # 9 months
        'F4': 13/12,     # 13 months
        'F5': 17/12      # 17 months
    }
    
    for col in futures_cols:
        if col in df.columns:
            try:
                prices_raw = pd.to_numeric(df[col], errors='coerce').values
                prices = prices_raw[valid_idx].copy()  # Apply same valid indices as spot
                
                if len(prices) == len(spot):
                    futures_list.append(prices)
                    maturities.append(maturity_dict[col])
                    print(f"  {col}: tau = {maturity_dict[col]:.3f} years ({maturity_dict[col]*12:.0f} months)")
            except Exception as e:
                print(f"  Warning: Could not load {col}: {e}")
                continue
    
    if not futures_list:
        print("Error: No futures data could be loaded")
        return
    
    futures = np.column_stack(futures_list)
    maturities = np.array(maturities)
    
    print(f"\n  Observations: {len(spot)}")
    print(f"  Spot price range: ${spot.min():.2f} - ${spot.max():.2f}")
    print(f"  Futures contracts loaded: {len(maturities)}")
    print(f"  Maturities: {maturities}")
    print(f"  Risk-free rate range: {risk_free_rates.min():.6f} - {risk_free_rates.max():.6f}")
    
    # Create and fit model
    model = GibsonSchwartz(spot, futures, maturities, risk_free_rates)
    
    print("\n" + "="*65)
    print("MAXIMUM LIKELIHOOD ESTIMATION")
    print("="*65)
    model.fit(verbose=True)
    model.print_parameters()
    
    # Extract states and plot
    print("\n" + "="*65)
    print("GENERATING RESULTS")
    print("="*65)
    spot_est, dy = model.extract_states()
    
    print("Generating plots...")
    model.plot_results()
    model.plot_futures()
    
    # Summary
    print("\n" + "="*65)
    print("SUMMARY STATISTICS")
    print("="*65)
    print(f"Spot price RMSE: ${np.sqrt(np.mean((spot - spot_est)**2)):.4f}/barrel")
    print(f"Convenience yield: {dy.min():.6f} to {dy.max():.6f}")
    print(f"Mean convenience yield: {dy.mean():.6f}")
    print("="*65 + "\n")


if __name__ == '__main__':
    main()
