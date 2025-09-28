from fenics import *
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# PARAMETERS
# -----------------------
L = 5.66         # mm
N = 300        # Finer mesh for accuracy
dt = 20        # s
t_end = 3600   # s
num_steps = int(t_end / dt)

# Physical parameters
c_hat_val = 0.011098*0.36  # mg/µL (0.01 * 0.36)
D = 0.83e-3         # mm²/s
A = 176.71         # mm²
rho = 0.0238607       # mg/µL
w = 0.36            # unitless

# -----------------------
# FEM SETUP
# -----------------------
mesh = IntervalMesh(N, 0, L)
V = FunctionSpace(mesh, "Lagrange", 1)

# Boundary condition
c_hat = Constant(c_hat_val)
bc = DirichletBC(V, c_hat, "near(x[0], 0)")

# Initial and trial/test functions
c = TrialFunction(V)
v = TestFunction(V)
c_n = Function(V)
c_nm1 = Function(V)
c_nm1.vector()[:] = 0.0

# Time tracking
Vc = [0.0]
flux_sum = 0.0

# -----------------------
# WEAK FORMULATION
# -----------------------
for n in range(num_steps):

    # Variational form (no reaction)
    a = (1/dt)*c*v*dx + D*dot(grad(c), grad(v))*dx
    Lform = (1/dt)*c_nm1*v*dx
    solve(a == Lform, c_n, bc)

    # Gradient at x=0
    grad_c = project(grad(c_n)[0], V)
    dc_dx = grad_c(Point(0.0))

    # Flux in (positive into domain)
    flux = max(0.0, -D * dc_dx)
    flux_sum += flux * dt

    # Volume absorbed
    Vc_current = (A / (w * rho)) * flux_sum
    Vc.append(Vc_current)

    # Step
    c_nm1.assign(c_n)

# -----------------------
# ANALYTICAL SOLUTION
# -----------------------
t_vals = np.linspace(0, t_end, num_steps + 1)
Vc_analytical = (2 * A * c_hat_val / (w * rho)) * np.sqrt(D * t_vals / np.pi)

# -----------------------
# PLOT RESULTS
# -----------------------
plt.plot(t_vals / 60, Vc, label='FEniCS (numerical)')
plt.plot(t_vals / 60, Vc_analytical, '--', label='Analytical')
plt.xlabel("Time (min)")
plt.ylabel("Vc (µL)")
plt.title("Accumulated Volume of CaCl2 Over Time (Simple Diffusion)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
