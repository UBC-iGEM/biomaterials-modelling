from fenics import *
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# FEM SETUP + SOLVER
# -----------------------
def constant_diffusion(params):
    L = params.L
    N = params.N 
    dt = params.dt
    t_end = params.t_end
    num_steps = params.num_steps
    c_CaCl2 = params.c_CaCl2
    w_CaCl2 = params.w_CaCl2
    c_hat = params.c_hat
    cA = params.cA
    D0 = params.D0
    K = params.K
    Nc = params.Nc
    A = params.A
    rho_CaCl2 = params.rho_CaCl2
    n = params.n
    alpha_gel = params.alpha_gel

    mesh = IntervalMesh(N, 0, L)
    V = FunctionSpace(mesh, "Lagrange", 1)

    def boundary_boolean_function(x, on_boundary):
        return on_boundary and near(x[0], 0)

    boundary_condition = DirichletBC(V, c_hat, boundary_boolean_function)

    u = TrialFunction(V)
    v = TestFunction(V)

    c_nm1 = Function(V)
    c_n = Function(V)
    c_nm1.vector()[:] = 0.0

    alpha_nm1 = Function(V)
    alpha_nm1.vector()[:] = 0.0
    alpha_n = Function(V)
    da_dt = Function(V)

    Vc = [0.0]
    flux_sum = 0.0

    for n in range(num_steps):
        # Update alpha using forward Euler
        alpha_rhs = project(K * (c_nm1 / cA) * (1 - alpha_nm1), V)
        alpha_n.vector()[:] = alpha_nm1.vector() + dt * alpha_rhs.vector()

        # Compute da/dt and reaction sink r
        da_dt.vector()[:] = (alpha_n.vector() - alpha_nm1.vector()) / dt
        r = project(Nc * cA * da_dt, V)

        # Solve variational problem for c_n
        a = (1 / dt) * u * v * dx + D0 * dot(grad(u), grad(v)) * dx
        l = (1 / dt) * c_nm1 * v * dx - r * v * dx
        solve(a == l, c_n, boundary_condition)

        # Compute dc/dx at x=0 using gradient
        grad_c = project(grad(c_n)[0], V)
        dc_dx = grad_c(Point(0.0))

        # Compute flux into domain
        flux = max(0.0, -float(D0) * dc_dx)
        flux_sum += flux * dt

        # Compute accumulated volume Vc
        Vc_current = (A / (w_CaCl2 * rho_CaCl2)) * flux_sum
        Vc.append(Vc_current)

        # Update for next time step
        c_nm1.assign(c_n)
        alpha_nm1.assign(alpha_n)

        print(f"Step {n:3d} | Time: {n * dt / 60:5.1f} min | Flux: {flux:.4e} mg/mm²/s | Vc: {Vc_current:.4e} µL")

    return Vc


def diffusion_alpha(params):
    L = params.L
    N = params.N 
    dt = params.dt
    t_end = params.t_end
    num_steps = params.num_steps
    c_CaCl2 = params.c_CaCl2
    w_CaCl2 = params.w_CaCl2
    c_hat = params.c_hat
    cA = params.cA
    D0 = params.D0
    D1 = params.D1
    K = params.K
    Nc = params.Nc
    A = params.A
    rho_CaCl2 = params.rho_CaCl2
    n = params.n
    alpha_gel = params.alpha_gel

    mesh = IntervalMesh(N, 0, L)
    V = FunctionSpace(mesh, "Lagrange", 1)

    def boundary_boolean_function(x, on_boundary):
        return on_boundary and near(x[0], 0)

    boundary_condition = DirichletBC(V, c_hat, boundary_boolean_function)

    u = TrialFunction(V)
    v = TestFunction(V)

    c_nm1 = Function(V)
    c_n = Function(V)
    c_nm1.vector()[:] = 0.0

    alpha_nm1 = Function(V)
    alpha_nm1.vector()[:] = 0.0
    alpha_n = Function(V)
    da_dt = Function(V)

    Vc = [0.0]
    flux_sum = 0.0

    for n in range(num_steps):
        # Update alpha using forward Euler
        alpha_rhs = project(K * (c_nm1 / cA) * (1 - alpha_nm1), V)
        alpha_n.vector()[:] = alpha_nm1.vector() + dt * alpha_rhs.vector()

        # Compute da/dt and reaction sink r
        da_dt.vector()[:] = (alpha_n.vector() - alpha_nm1.vector()) / dt
        r = project(Nc * cA * da_dt, V)

        #Compute diffusion coefficient D_alpha
        denominator = Constant(np.exp(-n * alpha_gel) - 1)
        D_alpha_expr = D0 + (D1 - D0) * (exp(-n * alpha_n / alpha_gel) - 1) / denominator
        D_alpha = project(D_alpha_expr, V)

        # Solve variational problem for c_n
        a = (1 / dt) * u * v * dx + D_alpha * dot(grad(u), grad(v)) * dx
        l = (1 / dt) * c_nm1 * v * dx - r * v * dx
        solve(a == l, c_n, boundary_condition)

        # Compute dc/dx at x=0 using gradient
        grad_c = project(grad(c_n)[0], V)
        dc_dx = grad_c(Point(0.0))

        # Compute flux into domain
        flux = max(0.0, -float(D_alpha(Point(0.0))) * dc_dx)
        flux_sum += flux * dt

        # Compute accumulated volume Vc
        Vc_current = (A / (w_CaCl2 * rho_CaCl2)) * flux_sum
        Vc.append(Vc_current)

        # Update for next time step
        c_nm1.assign(c_n)
        alpha_nm1.assign(alpha_n)

        print(f"Step {n:3d} | Time: {n * dt / 60:5.1f} min | Flux: {flux:.4e} mg/mm²/s | Vc: {Vc_current:.4e} µL")

    return Vc

