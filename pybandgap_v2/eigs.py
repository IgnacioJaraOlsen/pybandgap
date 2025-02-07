import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc

def calc_eigs(A, B=None, n_eigs=6, which="LM", tol=1e-7):
    """
    Calcula los eigenvalores y eigenvectores del problema generalizado Ax = λBx
    usando PETSc/SLEPc, similar a eigs() de MATLAB.

    Parámetros:
    -----------
    A : numpy.ndarray o petsc4py.PETSc.Mat
        Matriz principal
    B : numpy.ndarray o petsc4py.PETSc.Mat, opcional
        Matriz B para problema generalizado. Si es None, se resuelve Ax = λx
    n_eigs : int, opcional
        Número de eigenvalores a calcular (por defecto 6)
    which : str, opcional
        Criterio de selección de eigenvalores:
        - 'LM': Mayor magnitud (por defecto)
        - 'SM': Menor magnitud
        - 'LR': Mayor parte real
        - 'SR': Menor parte real
        - 'LI': Mayor parte imaginaria
        - 'SI': Menor parte imaginaria
        - 'TM': Magnitud objetivo (requiere establecer target)
        - 'TR': Parte real objetivo
        - 'TI': Parte imaginaria objetivo
    tol : float, opcional
        Tolerancia para la convergencia

    Retorna:
    --------
    eigenvals : numpy.ndarray
        Vector de eigenvalores
    eigenvects : numpy.ndarray
        Matriz con los eigenvectores como columnas
    """
    # Convertir A a formato PETSc si no lo es
    if not isinstance(A, PETSc.Mat):
        n = A.shape[0]
        matA = PETSc.Mat().createDense(size=(n, n))
        matA.setUp()

        for i in range(n):
            for j in range(n):
                matA.setValue(i, j, A[i, j])
        matA.assemble()
    else:
        matA = A

    # Convertir B a formato PETSc si no lo es
    matB = None
    if B is not None:
        if not isinstance(B, PETSc.Mat):
            n = B.shape[0]
            matB = PETSc.Mat().createDense(size=(n, n))
            matB.setUp()

            for i in range(n):
                for j in range(n):
                    matB.setValue(i, j, B[i, j])
            matB.assemble()
        else:
            matB = B

    # Crear el solver de eigenvalores
    E = SLEPc.EPS()
    E.create()

    # Configurar el problema
    if matB is not None:
        E.setOperators(matA, matB)
        E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    else:
        E.setOperators(matA)
        E.setProblemType(SLEPc.EPS.ProblemType.NHEP)

    # Configurar el criterio de selección
    which_dict = {
        "LM": SLEPc.EPS.Which.LARGEST_MAGNITUDE,
        "SM": SLEPc.EPS.Which.SMALLEST_MAGNITUDE,
        "LR": SLEPc.EPS.Which.LARGEST_REAL,
        "SR": SLEPc.EPS.Which.SMALLEST_REAL,
        "LI": SLEPc.EPS.Which.LARGEST_IMAGINARY,
        "SI": SLEPc.EPS.Which.SMALLEST_IMAGINARY,
        "TM": SLEPc.EPS.Which.TARGET_MAGNITUDE,
        "TR": SLEPc.EPS.Which.TARGET_REAL,
        "TI": SLEPc.EPS.Which.TARGET_IMAGINARY,
    }

    E.setWhichEigenpairs(which_dict[which])

    # Configurar parámetros del solver
    E.setDimensions(n_eigs)
    E.setTolerances(tol)

    # Resolver el problema de eigenvalores
    E.solve()

    # Obtener el número de eigenvalores convergidos
    nconv = E.getConverged()
    if nconv < n_eigs:
        print(f"Advertencia: Solo {nconv} de {n_eigs} eigenvalores convergieron")

    # Extraer eigenvalores y eigenvectores
    eigenvals = np.zeros(n_eigs, dtype=complex)
    eigenvects = np.zeros((matA.getSize()[0], n_eigs), dtype=complex)

    vr, vi = matA.createVecs()

    for i in range(n_eigs):
        k = E.getEigenpair(i, vr, vi)
        eigenvals[i] = k.real + 1j * k.imag

        vr_array = vr.getArray()
        vi_array = vi.getArray()
        eigenvects[:, i] = vr_array + 1j * vi_array

    if B is not None:
        eigenvects = gram_schmidt_B(eigenvects, B)
        eigenvects = normalize_eigenvectors_B(eigenvects, B)

    return eigenvals, eigenvects

def normalize_eigenvectors_B(eigenvects, B):
    """
    Normaliza los eigenvectores usando la matriz B y los hace B-ortonormales.

    Parámetros:
    -----------
    eigenvects : numpy.ndarray
        Matriz con los eigenvectores como columnas.
    B : numpy.ndarray
        Matriz para normalización B-ortonormal.

    Retorna:
    --------
    normalized_eigenvects : numpy.ndarray
        Matriz con los eigenvectores B-ortonormales como columnas.
    """
    B = B[:,:]
    
    n_eigs = eigenvects.shape[1]
    normalized_eigenvects = np.zeros_like(eigenvects, dtype=complex)

    for i in range(n_eigs):
        v = eigenvects[:, i]
        norm_B = np.sqrt(np.conj(v).T @ B @ v)
        normalized_eigenvects[:, i] = v / norm_B

    # Verificar ortonormalidad y ajustar si es necesario
    for i in range(n_eigs):
        for j in range(i, n_eigs):
            dot_product = np.conj(normalized_eigenvects[:, i]).T @ B @ normalized_eigenvects[:, j]
            if i == j:
                assert np.isclose(dot_product, 1.0, atol=1e-8), f"Vector {i} no está correctamente normalizado"
            else:
                assert np.isclose(dot_product, 0.0, atol=1e-8), f"Vectores {i} y {j} no son ortogonales"

    return normalized_eigenvects


def gram_schmidt_B(vectors, B):
    """
    Ortonormaliza un conjunto de vectores según la métrica inducida por B.

    Parámetros:
    -----------
    vectors : numpy.ndarray
        Matriz cuyas columnas son los vectores a ortonormalizar.
    B : numpy.ndarray
        Matriz para definir la métrica B.

    Retorna:
    --------
    ortonormal_vectors : numpy.ndarray
        Matriz con los vectores B-ortonormales como columnas.
    """
    B = B[:,:]
    n_vectors = vectors.shape[1]
    n_dim = vectors.shape[0]
    ortonormal_vectors = np.zeros((n_dim, n_vectors), dtype=complex)

    for i in range(n_vectors):
        # Proyección y ortogonalización respecto a los vectores anteriores
        v = vectors[:, i]
        for j in range(i):
            v_proj = np.conj(ortonormal_vectors[:, j]).T @ B @ v
            v -= v_proj * ortonormal_vectors[:, j]

        # Normalización según la norma inducida por B
        norm_B = np.sqrt(np.conj(v).T @ B @ v)
        if norm_B < 1e-12:  # Verificar si el vector se anuló numéricamente
            raise ValueError(f"El vector {i} es linealmente dependiente de los anteriores.")
        ortonormal_vectors[:, i] = v / norm_B

    return ortonormal_vectors