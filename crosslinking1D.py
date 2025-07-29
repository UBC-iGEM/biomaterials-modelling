from fenics import *
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# PARAMETERS AND CONSTANTS
# -----------------------
L = 28  # mm — Domain length
N = 264  # Number of spatial elements
dt = 20  # s — Time step
t_end = 1200  # s — Total time
num_steps = int(t_end / dt) 

# Concentration and reaction parameters
c_CaCl2 = 0.01      # mg/µL
w_CaCl2 = 0.36
c_hat = Constant(c_CaCl2 * w_CaCl2)
cA = Constant(0.08) # mg/µL

D0 = Constant(0.83e-3)  # mm²/s
K = Constant(0.03)      # 1/s
Nc = Constant(0.1)      # dimensionless

# Volume calculation constants
A = 17.81  # mm²
rho_CaCl2 = 0.0215  # mg/µL

# -----------------------
# FEM SETUP
# -----------------------
mesh = IntervalMesh(N, 0, L)
V = FunctionSpace(mesh, "Lagrange", 1)

def boundary_boolean_function(x, on_boundary):
    return on_boundary and near(x[0], 0)

boundary_condition = DirichletBC(V, c_hat, boundary_boolean_function)

c = TrialFunction(V)
v = TestFunction(V)

c_nm1 = Function(V)
c_n = Function(V)
c_nm1.vector()[:] = 0.0

alpha_nm1 = Function(V)
alpha_nm1.vector()[:] = 0.0
alpha_n = Function(V)
da_dt = Function(V)

# -----------------------
# TIME STEPPING
# -----------------------
Vc = []
Vc.append(0.0)
flux_sum = 0.0

for n in range(num_steps):
    # Update alpha using forward Euler
    alpha_rhs = project(K * (c_nm1 / cA) * (1 - alpha_nm1), V)
    alpha_n.vector()[:] = alpha_nm1.vector() + dt * alpha_rhs.vector()

    # Compute da/dt and reaction sink r
    da_dt.vector()[:] = (alpha_n.vector() - alpha_nm1.vector()) / dt
    r = project(Nc * cA * da_dt, V)

    # Solve variational problem for c_n
    a = (1/dt)*c*v*dx + D0*dot(grad(c), grad(v))*dx
    l = (1/dt)*c_nm1*v*dx - r*v*dx 
    solve(a == l, c_n, boundary_condition)
 
    # Compute dc/dx at x=0 using forward difference
    grad_c = project(grad(c_n)[0], V)
    dc_dx = grad_c(Point(0.0))
    print(f"c(0) = {c_n(Point(0.0)):.4f}, expected = {float(c_hat):.4f}")

    # Compute flux into domain
    flux = max(0.0, -float(D0) * dc_dx)
    flux_sum += flux * dt

    # Compute accumulated volume Vc
    Vc_current = (A / (w_CaCl2 * rho_CaCl2)) * flux_sum
    Vc.append(Vc_current)

    # Update for next step
    c_nm1.assign(c_n)
    alpha_nm1.assign(alpha_n)

    print(f"Step {n:3d} | Time: {n*dt/60:5.1f} min | Flux: {flux:.4e} mg/mm²/s | Vc: {Vc_current:.4e} µL")

# -----------------------
# PLOT RESULTS
# -----------------------

time_points = np.linspace(0, t_end, num_steps + 1) / 60  # min

plt.figure()
plt.plot(time_points, Vc)
plt.xlabel("Time (minutes)")
plt.ylabel("V_c(t) (µL)")
plt.title("Accumulated Volume Over Time")
plt.grid(True)
plt.tight_layout()

plt.show()
