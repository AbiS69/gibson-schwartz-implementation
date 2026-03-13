import pandas as pd
import numpy as np
import statsmodels.api as sm
from params import GibsonSchwartzParams

def get_calibrated_parameters(file_path):
    # 1. Chargement de ta data Excel propre
    df = pd.read_excel(file_path)
    
    # On s'assure que c'est trié par date
    df = df.sort_values('Exchange Date').reset_index(drop=True)

    # 2. Calcul du Convenience Yield instantané (delta)
    # Formule : delta = r - ln(F2/F1) / (tau2 - tau1)
    tau1, tau2 = 1/12, 2/12  # Maturités 1 mois et 2 mois
    df['delta'] = df['r_1m'] - np.log(df['Close LCOc2'] / df['Close LCOc1']) / (tau2 - tau1)

    # 3. Calcul de la cible long-terme (delta_bar_q)
    # On utilise le contrat lointain LCOh9 (proxy 5 ans environ)
    tau_h9 = 5.0 # À ajuster selon la maturité réelle de ton H9
    df['delta_star_series'] = df['r_5y'] - np.log(df['Close LCOh9'] / df['Close LCOc1']) / (tau_h9 - tau1)
    delta_bar_q = df['delta_star_series'].mean()

    # 4. Régression pour Kappa (Vitesse de retour)
    # Y = variation journalière, X = niveau de la veille
    df['delta_diff'] = df['delta'].diff()
    df['delta_lag'] = df['delta'].shift(1)
    
    reg_data = df[['delta_diff', 'delta_lag']].dropna()
    
    Y = reg_data['delta_diff']
    X = sm.add_constant(reg_data['delta_lag'])
    
    model = sm.OLS(Y, X).fit()
    a, b = model.params['const'], model.params['delta_lag']
    
    kappa = -b * 252 # Annualisation
    # alpha_implied = a / -b  # Optionnel : alpha calculé par la régression

    # 5. Calcul des volatilités (sigma)
    # Sigma_delta : écart-type des résidus de la régression
    sigma_delta = model.resid.std() * np.sqrt(252)
    
    # Sigma_S : volatilité du prix spot (LCOc1)
    df['spot_ret'] = np.log(df['Close LCOc1'] / df['Close LCOc1'].shift(1))
    sigma_s = df['spot_ret'].std() * np.sqrt(252)

    # 6. Corrélation (rho)
    rho = df['spot_ret'].corr(df['delta_diff'])

    # On renvoie l'objet avec tous les paramètres
    return GibsonSchwartzParams(
        r=df['r_1m'].iloc[-1], # On prend le dernier taux connu pour le pricing
        kappa=kappa,
        delta_bar_q=delta_bar_q,
        sigma_s=sigma_s,
        sigma_delta=sigma_delta,
        rho=rho
    )
