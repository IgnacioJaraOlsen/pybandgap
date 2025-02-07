import numpy as np
from scipy.sparse.linalg import eigs, ArpackNoConvergence
from tabulate import tabulate
from slepc_eigs import calc_eigs
from petsc4py import PETSc
from scipy.sparse import csc_matrix
import time  # Importamos la librería time para medir el tiempo de ejecución

# Funciones de conversión y cálculo
def _convert_vector(vec, backend):
    """Convierte un vector al tipo del backend especificado"""
    if backend == "petsc" and not isinstance(vec, PETSc.Vec):
        return PETSc.Vec().createWithArray(vec)
    elif backend == "numpy" and isinstance(vec, PETSc.Vec):
        return vec.getArray()
    return vec

def _convert_matrix(mat, backend):
    """Convierte una matriz al tipo del backend especificado"""
    if backend == "petsc" and not isinstance(mat, PETSc.Mat):
        mat_np = np.asarray(mat)
        petsc_mat = PETSc.Mat().createDense(mat_np.shape)
        petsc_mat.setUp()
        petsc_mat.setValues(range(mat_np.shape[0]), range(mat_np.shape[1]), mat_np)
        petsc_mat.assemble()
        return petsc_mat
    elif backend == "numpy" and isinstance(mat, PETSc.Mat):
        return mat.getDenseArray()
    return np.asarray(mat) if backend == "numpy" else mat

def _compute_numerator(mu, u, dK_dx, dM_dx, backend):
    """Calcula el numerador de la derivada del valor propio"""
    if backend == "petsc":
        temp_mat = dK_dx.copy()
        temp_mat.axpy(-mu, dM_dx)
        Au = temp_mat.createVecLeft()
        temp_mat.mult(u, Au)
        return u.dot(Au)
    else:
        return np.conj(u).T @ (dK_dx - mu * dM_dx) @ u

def _compute_denominator(u, M, backend):
    """Normalización M-ortonormal"""
    if backend == "petsc":
        Mu = M.createVecLeft()
        M.mult(u, Mu)
        denom = u.dot(Mu)
        u.scale(1.0 / np.sqrt(denom))  # Normalización explícita
        return 1.0
    else:
        denom = np.conj(u).T @ M @ u
        u /= np.sqrt(denom)
        return 1.0

def compute_eigen_derivative(mu, u, dK_dx, dM_dx, M, backend="numpy"):
    """
    Calcula la derivada de valores propios para problemas de vibración/buckling
    """
    u = _convert_vector(u, backend)
    dK_dx = _convert_matrix(dK_dx, backend)
    dM_dx = _convert_matrix(dM_dx, backend)
    M = _convert_matrix(M, backend)

    numerator = _compute_numerator(mu, u, dK_dx, dM_dx, backend)
    denominator = _compute_denominator(u, M, backend)

    return numerator / denominator

def compare_derivatives(K, M, dK_dx, dM_dx, n_eigs=2):
    """
    Compara las derivadas entre SciPy y SLEPc con manejo de convergencia
    """
    # Configuración para SciPy
    K_sparse = csc_matrix(K)
    M_sparse = csc_matrix(M)
    
    # Medir tiempo de ejecución para SciPy
    start_time_scipy = time.time()
    try:
        eigenvals_scipy, eigenvects_scipy = eigs(
            K_sparse,
            k=n_eigs,
            M=M_sparse,
            sigma=0,  # Búsqueda alrededor del origen
            which="LM",  # Largest Magnitude relativo a sigma
            maxiter=5000,
            tol=1e-12,
            v0=np.random.rand(K.shape[0])  # Vector inicial aleatorio
        )
    except ArpackNoConvergence as e:
        print(f"Advertencia SciPy: {e}")
        eigenvals_scipy = e.eigenvalues
        eigenvects_scipy = e.eigenvectors
    end_time_scipy = time.time()
    tiempo_scipy = end_time_scipy - start_time_scipy

    # Configuración para SLEPc
    start_time_slepc = time.time()
    eigenvals_slepc, eigenvects_slepc = calc_eigs(
        K,
        M,
        n_eigs=n_eigs,
        which="SM",
        tol=1e-12
    )
    end_time_slepc = time.time()
    tiempo_slepc = end_time_slepc - start_time_slepc

    # Ordenamiento
    def sort_eigenpairs(eigenvals, eigenvects):
        sorted_indices = np.argsort(np.abs(eigenvals))
        return eigenvals[sorted_indices], eigenvects[:, sorted_indices]

    eigenvals_scipy, eigenvects_scipy = sort_eigenpairs(eigenvals_scipy, eigenvects_scipy)
    eigenvals_slepc, eigenvects_slepc = sort_eigenpairs(eigenvals_slepc, eigenvects_slepc)

    # Comparación detallada
    table_data = []
    for i in range(min(n_eigs, len(eigenvals_scipy))):  # Manejar casos de convergencia parcial
        mu_scipy = eigenvals_scipy[i]
        u_scipy = eigenvects_scipy[:, i]
        deriv_scipy = compute_eigen_derivative(mu_scipy, u_scipy, dK_dx, dM_dx, M)

        mu_slepc = eigenvals_slepc[i]
        u_slepc = eigenvects_slepc[:, i]
        deriv_slepc = compute_eigen_derivative(mu_slepc, u_slepc, dK_dx, dM_dx, M, backend="petsc")

        diff_eig = abs(mu_scipy - mu_slepc)
        diff_deriv = abs(deriv_scipy - deriv_slepc)

        format = ".3e"
        
        table_data.append([
            i,
            f"{mu_scipy:{format}}",
            f"{deriv_scipy:{format}}",
            f"{mu_slepc:{format}}",
            f"{deriv_slepc:{format}}",
            f"{diff_eig:.2e}",
            f"{diff_deriv:.2e}"
        ])

    headers = [
        "Índice", "Valor (SciPy)", "Deriv (SciPy)", 
        "Valor (SLEPc)", "Deriv (SLEPc)", "Δ Valor", "Δ Deriv"
    ]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Mostrar tiempos de ejecución
    print(f"\nTiempo de ejecución SciPy: {tiempo_scipy:.4f} segundos")
    print(f"Tiempo de ejecución SLEPc: {tiempo_slepc:.4f} segundos")

if __name__ == "__main__":    
    print("=== Comparación SciPy vs SLEPc ===")
    # Matrices de prueba hermíticas para mejor convergencia
    K_test = np.array([
        [5, 1, 0, 0, 0],
        [1, 4, 2, 0, 0],
        [0, 2, 3, 1, 0],
        [0, 0, 1, 6, 3],
        [0, 0, 0, 3, 7]
    ], dtype=complex)  # Matriz simétrica real
    
    M_test = np.diag([2, 3, 4, 5, 6]).astype(float)
    
    # Convertir a complejo sin componentes imaginarios
    K_test = K_test.astype(complex)
    M_test = M_test.astype(complex)
    
    dK_dx_test = 0.01 * K_test
    dM_dx_test = np.zeros_like(M_test)
    
    compare_derivatives(K_test, M_test, dK_dx_test, dM_dx_test, n_eigs=3)