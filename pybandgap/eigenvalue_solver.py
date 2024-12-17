from typing import List, Tuple

import dolfinx.fem as fem
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc
from dolfinx.fem import FunctionSpace


def solve_generalized_eigenvalue_problem(
    A: PETSc.Mat,
    B: PETSc.Mat,
    nev: int = 10,
    tol: float = 1e-12,
    max_it: int = 20,
    target: float = 0.0,
) -> SLEPc.EPS:
    
    # Create MPI communicator
    comm = PETSc.COMM_WORLD
    
    # Create eigenvalue solver
    eps = SLEPc.EPS().create(comm=comm)
    eps.setOperators(A, B)
    
    # Set problem type to generalized
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)  # Non-Hermitian generalized
    
    # # Set the spectral transformation to target small magnitude eigenvalues
    # st = eps.getST()
    # st.setType(SLEPc.ST.Type.SINVERT)  # Shift-and-invert
    
    # Set solver parameters
    eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
    # eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    # eps.setTarget(target)  # Target small magnitude
    
    # Set solver type
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    
    # Set number of eigenvalues to compute
    eps.setDimensions(nev=nev)
    
    # Optional: set solver tolerances
    eps.setTolerances(max_it=max_it, tol=tol)
    
    # Solve the eigenvalue problem
    eps.solve()

    # Retrieve eigenvalues
    eigenvalues = []
    for i in range(eps.getConverged()):
        eigenvalue = eps.getEigenvalue(i)
        eigenvalues.append(eigenvalue.real)
    
    return eigenvalues
