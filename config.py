from fenics import *
from dataclasses import dataclass
import numpy as np

# # -----------------------
# # PARAMETERS
# # -----------------------
# L = 28  # mm — Domain length
# N = 264  # Number of spatial elements
# dt = 20  # s — Time step
# t_end = 1200  # s — Total time
# num_steps = int(t_end / dt)

# # Concentration and reaction parameters
# c_CaCl2 = 0.01      # mg/µL
# w_CaCl2 = 0.36
# c_hat = Constant(c_CaCl2 * w_CaCl2)
# cA = Constant(0.08) # mg/µL

# D0 = Constant(0.83e-3)  # mm²/s
# D1 = Constant(0.415e-3)   # mm²/s
# K = Constant(0.03)      # 1/s
# Nc = Constant(0.1)      # dimensionless

# # Volume calculation constants
# A = 17.81  # mm²
# rho_CaCl2 = 0.0215  # mg/µL

# #D_alpha parameters
# n = 5
# alpha_gel = 0.2


@dataclass
class Parameters:
    L: float
    N: int
    dt: float
    t_end: float
    c_CaCl2: float
    w_CaCl2: float
    cA: Constant
    D0: Constant
    D1: Constant
    K: Constant
    Nc: Constant
    A: float
    rho_CaCl2: float
    n: int
    alpha_gel: float

    @property
    def num_steps(self) -> int:
        return int(self.t_end / self.dt)
    
    @property
    def c_hat(self) -> Constant:
        return Constant(self.c_CaCl2 * self.w_CaCl2)

paper_params = Parameters(
    L=28.0,
    N=264,
    dt=20.0,
    t_end=1200.0,
    c_CaCl2=0.01,
    w_CaCl2=0.36,
    cA=Constant(0.08),
    D0=Constant(0.83e-3),
    D1=Constant(0.415e-3),
    K=Constant(0.03),
    Nc=Constant(0.1),
    A=17.81,
    rho_CaCl2=0.0215,
    n=5,
    alpha_gel=0.2
)

alginate_only_params = Parameters(
    L=5.66, #edited mm
    N=300,
    dt=20.0,
    t_end=3600.0, #edited
    c_CaCl2= 0.011098, #edited (mg/µL)
    w_CaCl2=0.36, #no need to change (mg/µL)
    cA=Constant(0.03),#edited 
    D0=Constant(0.83e-3), #no need to change
    D1=Constant(0.415e-3),#no need to change
    K=Constant(0.03),
    Nc=Constant(0.1),
    A=176.71, #edited (mm^2 = pi*(7mm/2)^2)
    rho_CaCl2= 0.0238607,#edited
    n=5,
    alpha_gel=0.2
)