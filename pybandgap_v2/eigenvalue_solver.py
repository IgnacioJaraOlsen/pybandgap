import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

def solve_generalized_eigenvalue_problem(
    A: PETSc.Mat,
    B: PETSc.Mat,
    nev: int = 10,
    tol: float = 1e-20,
    max_it: int = 1000,
):
    comm = PETSc.COMM_WORLD
    eps = SLEPc.EPS().create(comm=comm)
    eps.setOperators(A, B)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setDimensions(nev=nev)
    eps.setTolerances(max_it=max_it, tol=tol)
    eps.solve()

    # Get the solution
    eigenvalues = np.zeros(nev, dtype= np.complex128)
    eigenvectors = np.zeros((nev,A.size[0]), dtype= np.complex128)
    
    vr, _ = A.createVecs()
    
    for i in range(nev):
        eigenvalue = eps.getEigenvalue(i)
        eps.getEigenvector(i, vr)
        
        # Convert vector to numpy array
        vec_np = vr.array.copy()  # Important: make a copy
        
        # Normalize to unit length (like SciPy)
        vec_np = vec_np / np.linalg.norm(vec_np)
        
        # Hacer que el componente más grande sea positivo
        max_idx = np.argmax(np.abs(vec_np))
        if vec_np[max_idx] < 0:
            vec_np = -vec_np
            
        eigenvalues[i] = eigenvalue.real
        eigenvectors[i,:] = vec_np
            
    return np.array(eigenvalues), eigenvectors