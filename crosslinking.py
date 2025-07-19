from dolfin import *

# Create a unit square mesh with 32x32 divisions
mesh = UnitSquareMesh(32, 32)

# Define function space: continuous piecewise linear functions
V = FunctionSpace(mesh, "Lagrange", 1)

# Define boundary condition: u = 0 on boundary
u_D = Constant(0.0)
bc = DirichletBC(V, u_D, "on_boundary")

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)  # Right-hand side of the equation
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# Solve the linear system
u = Function(V)
solve(a == L, u, bc)

# Plot the solution
import matplotlib.pyplot as plt
plot(u)
plt.title("Solution to Poisson's Equation")
plt.show()