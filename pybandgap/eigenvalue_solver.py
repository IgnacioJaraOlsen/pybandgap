from typing import List, Tuple

import dolfinx.fem as fem
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

def solve_generalized_eigenvalue_problem(
    A: PETSc.Mat,
    B: PETSc.Mat,
    nev: int = 10,
    tol: float = 1e-20,
    max_it: int = 1000,
):
    global test
    
    # Create MPI communicator
    comm = PETSc.COMM_WORLD
    
    # Create eigenvalue solver
    eps = SLEPc.EPS().create(comm=comm)
    eps.setOperators(A, B)
    
    # Set problem type to generalized
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)  # Non-Hermitian generalized
    
    # Set solver parameters
    eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
    
    # Set solver type
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    
    # Set number of eigenvalues to compute
    eps.setDimensions(nev=nev)
    
    # Optional: set solver tolerances
    eps.setTolerances(max_it=max_it, tol=tol)
    
    # Solve the eigenvalue problem
    eps.solve()
    
    vr, _ = A.getVecs()
    
    # Retrieve eigenvalues
    eigenvalues = []
    eigenvectors = []
    for i in range(eps.getConverged()):
        eigenvalue = eps.getEigenpair(i, vr)
        eigenvalues.append(eigenvalue)
        eigenvectors.append(vr)
            
    return eigenvalues, eigenvectors
