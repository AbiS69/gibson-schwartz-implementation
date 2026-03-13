
from dataclasses import dataclass

@dataclass(frozen=True)
class GibsonSchwartzParams:
    r: float            # Taux sans risque
    kappa: float        # Vitesse de retour à la moyenne
    delta_bar_q: float  # Cible long terme (alpha)
    sigma_s: float      # Volatilité du spot
    sigma_delta: float  # Volatilité du yield
    rho: float          # Corrélation
