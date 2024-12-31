import ufl
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_interval, create_rectangle, CellType
from dolfinx.fem import form, assemble_scalar, functionspace, Constant, Function
from dolfinx.fem.petsc import create_vector, create_matrix, assemble_vector, assemble_matrix
from petsc4py import PETSc

mesh = create_interval(MPI.COMM_WORLD, 1, points=(0, 1))
A1 = np.pi*4e-6
A2 = np.pi*16e-6

V = functionspace(mesh, ("CG", 1, (2,1)))
V0 = functionspace(mesh, ("DG", 0))

x_A = Function(V0)
x_A.x.array[:] = 0
A = (A2 - A1) * x_A + A1

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = A*ufl.inner(u, v)*ufl.dx

da_form = form(ufl.derivative(a, x_A))
dA = assemble_vector(da_form)
dA.assemble()

print(dA[:])

A = assemble_matrix(form(a))
A.assemble()
print(A[:,:])