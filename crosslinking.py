from fenics import *
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# PARAMETERS AND CONSTANTS
# -----------------------

L = 28  # mm — Domain length (28 mm as in the paper)
N = 64  # Number of spatial elements
dt = 20  # s — Time step size
t_end = 1200  # s — Total simulation time
num_steps = int(t_end / dt)

# Concentration parameters
c_CaCl2 = 0.01  # mg/µL — CaCl₂ preparation concentration
w_CaCl2 = 0.36  # unitless — weight fraction of Ca²⁺ in CaCl₂
c_hat = Constant(c_CaCl2 * w_CaCl2)  # mg/µL — Boundary concentration of Ca²⁺
cA = Constant(0.08)  # mg/µL — Initial concentration of polymer binding sites

# Diffusion and reaction
D0 = Constant(0.83e-3)  # mm²/s — Diffusion coefficient
K = Constant(0.03)      # 1/s — Reaction rate constant
Nc = Constant(0.1)      # dimensionless — Reaction yield coefficient

# Volume tracking constants
A = 17.81  # mm² — Cross-sectional area of hydrogel (from paper)
rho_CaCl2 = 0.0215  # mg/µL — Density of CaCl₂ solution

# -----------------------
# FEM SETUP
# -----------------------

# Mesh and function space
mesh = IntervalMesh(N, 0, L)
V = FunctionSpace(mesh, "Lagrange", 1)

# Dirichlet boundary condition at x = 0
def boundary_boolean_function(x, on_boundary):
    return on_boundary and near(x[0], 0)

boundary_condition = DirichletBC(V, c_hat, boundary_boolean_function)

# Trial and test functions
c = TrialFunction(V)
v = TestFunction(V)

# Concentration and alpha functions
c_nm1 = Function(V)   # c at previous step
c_n = Function(V)     # c at current step
c_nm1.vector()[:] = 0.0 #Initial Condition

alpha_nm1 = Function(V)  # α at previous step
alpha_nm1.vector()[:] = 0.0
alpha_n = Function(V)    # α at current step
da_dt = Function(V)      # ∂α/∂t

# -----------------------
# TIME STEPPING
# -----------------------

Vc = []           # Accumulated volume in µL
flux_sum = 0.0    # Integrated flux over time (mg/mm²)

for n in range(num_steps):
    # Update alpha (conversion) using forward Euler
    alpha_rhs = project(K * (c_nm1 / cA) * (1 - alpha_nm1), V)
    alpha_n.vector()[:] = alpha_nm1.vector() + dt * alpha_rhs.vector()

    # Compute ∂α/∂t
    da_dt.vector()[:] = (alpha_n.vector() - alpha_nm1.vector()) / dt

    # Source term: reaction rate r = Nc * cA * ∂α/∂t
    r = project(Nc * cA * da_dt, V)

    # Variational problem: solve for c_n
    a = (1/dt)*c*v*dx + D0*dot(grad(c), grad(v))*dx
    l = (1/dt)*c_nm1*v*dx + r*v*dx
    solve(a == l, c_n, boundary_condition)

    # Estimate ∂c/∂x at x = 0 using forward finite difference
    x0 = 0.0
    x1 = mesh.coordinates()[1][0]
    c0 = c_n(Point(x0))
    c1 = c_n(Point(x1))
    dc_dx = (c1 - c0) / (x1 - x0)

    # Compute flux into the domain (mg/mm²/s)
    flux = -float(D0) * dc_dx
    
    # Accumulate flux over time (to get mg/mm²)
    flux_sum += flux * dt

    # Compute accumulated volume Vc (µL)
    Vc_current = (A / (w_CaCl2 * rho_CaCl2)) * flux_sum  # Units: µL
    Vc.append(Vc_current)

    # Update variables for next time step
    c_nm1.assign(c_n)
    alpha_nm1.assign(alpha_n)

    # Diagnostics
    print(f"Step {n:4d} | Time: {n*dt/60:5.1f} min | Flux: {flux:.4e} mg/mm²/s | Vc: {Vc[-1]:.4e} µL")

# -----------------------
# PLOT RESULTS
# -----------------------

time_points = np.linspace(0, t_end, num_steps) / 60  # min

plt.figure()
plt.plot(time_points, Vc)
plt.xlabel("Time (minutes)")
plt.ylabel("V_c(t) (µL)")
plt.title("Accumulated Volume Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()
