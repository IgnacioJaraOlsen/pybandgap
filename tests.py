import numpy as np
from petsc4py import PETSc
from pybandgap_v2.eigs import eigs
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.io import loadmat

matrices = {
    1: {
        'A': np.array([[4, 1], [1, 3]]),
        'B': np.array([[2, 0], [0, 2]])
    },
    2: {
        'A': np.array([[1, 2, 0], [2, 1, 0], [0, 0, 1]]),
        'B': np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
    },
    3: {
        'A': np.array([[2, 1], [1, 2]]),
        'B': np.array([[3, 0], [0, 3]])
    },
    4: {
        'A': np.eye(3),
        'B': np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]])
    },
    5: {
        'A': np.array([[3, -1], [-1, 3]]),
        'B': np.eye(2)
    },
    6: {
        'A': np.array([[5, 2], [2, 5]]),
        'B': np.array([[1, 0], [0, 2]])
    },
    7: {
        'A': np.array([[6, 4], [4, 6]]),
        'B': np.array([[1, 0], [0, 2]])
    },
    8: {
        'A': np.array([[1, 1], [1, 2]]),
        'B': np.array([[2, 0], [0, 1]])
    },
    9: {
        'A': np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
        'B': np.array([[1, 0, 0], [0, 2+1j, 0], [0, 0, 3]])
    },
    10: {
        'A': np.array([[1+1j, 2], [3, 4]]),
        'B': np.eye(2)
    }
}

def convert_to_petsc(A, B):
    """
    Convierte matrices NumPy a formato PETSc con mayor precisión
    """
    n = A.shape[0]
    A_petsc = PETSc.Mat().createDense([n, n])
    B_petsc = PETSc.Mat().createDense([n, n])
    
    # Configurar para mayor precisión
    A_petsc.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    B_petsc.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    
    A_petsc.setValues(range(n), range(n), A)
    B_petsc.setValues(range(n), range(n), B)
    
    A_petsc.assemble()
    B_petsc.assemble()
    
    return A_petsc, B_petsc


def plot_comparison_eigenvalues_vectors(eigenvalues_list, eigenvectors_list, labels):
    """
    Compara diferentes funciones de eigs mostrando los valores propios y vectores propios.
    Los vectores propios se ajustan para que siempre apunten hacia la derecha.
    
    :param eigenvalues_list: List of arrays, each containing the eigenvalues from a function.
    :param eigenvectors_list: List of arrays, each containing the eigenvectors from a function.
    :param labels: List of strings, each representing the label for the corresponding function.
    """
    # Número de funciones a comparar
    num_functions = len(eigenvalues_list)
    
    # Crear la figura y los ejes para la visualización
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    
    # Colores para cada función
    colors = plt.cm.get_cmap("tab10", num_functions)
    
    # Gráfico de los valores propios
    for i in range(num_functions):
        eigenvalues = eigenvalues_list[i]
        ax[0].scatter(np.real(eigenvalues), np.imag(eigenvalues), 
                     label=labels[i], s=100, marker=f"${i+1}$", linewidths=1)
    
    ax[0].set_title("Comparación de Eigenvalues")
    ax[0].set_xlabel("Parte Real")
    ax[0].set_ylabel("Parte Imaginaria")
    ax[0].axhline(0, color='black', linewidth=0.5)
    ax[0].axvline(0, color='black', linewidth=0.5)
    ax[0].legend()
    
    # Gráfico de los vectores propios
    for i in range(num_functions):
        eigenvectors = eigenvectors_list[i]
        color = colors(i)
        
        for j in range(eigenvectors.shape[1]):
            vector = eigenvectors[:, j]
            
            # Calcular el ángulo del vector con respecto al eje x
            angle = np.angle(vector[0])
            
            # Si el vector apunta hacia la izquierda (ángulo > 90° o < -90°),
            # multiplicarlo por -1 para que apunte hacia la derecha
            if abs(angle) > np.pi/2:
                vector = -vector
                
            # Dibujar la flecha
            ax[1].arrow(0, 0,
                        np.real(vector[0]), np.imag(vector[0]),
                        head_width=0.05, head_length=0.1,
                        fc=color, ec=color, alpha=0.6,
                        length_includes_head=True,
                        label=labels[i] if j == 0 else "")
            
            # Agregar un punto en la punta de la flecha
            ax[1].scatter(np.real(vector[0]), np.imag(vector[0]), 
                          color=color, s=50, marker=f"${i+1}$")
    
    ax[1].set_title("Comparación de Eigenvectors")
    ax[1].set_xlabel("Parte Real")
    ax[1].set_ylabel("Parte Imaginaria")
    ax[1].axis('equal')
    ax[1].axhline(0, color='black', linewidth=0.5)
    ax[1].axvline(0, color='black', linewidth=0.5)
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()
def plot_comparison_eigenvalues_vectors(eigenvalues_list, eigenvectors_list, labels):
    """
    Compara diferentes funciones de eigs mostrando los valores propios y vectores propios.
    Los vectores propios se ajustan para que siempre apunten hacia la derecha.
    
    :param eigenvalues_list: List of arrays, each containing the eigenvalues from a function.
    :param eigenvectors_list: List of arrays, each containing the eigenvectors from a function.
    :param labels: List of strings, each representing the label for the corresponding function.
    """
    # Número de funciones a comparar
    num_functions = len(eigenvalues_list)
    
    # Crear la figura y los ejes para la visualización
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    
    # Colores para cada función
    colors = plt.cm.get_cmap("Set1")#, num_functions)
    
    # Gráfico de los valores propios
    for i in range(num_functions):
        eigenvalues = eigenvalues_list[i]
        ax[0].scatter(np.real(eigenvalues), np.imag(eigenvalues), 
                     label=labels[i], s=100, marker=f"{i+1}", linewidths=1, color = colors(i))
    
    ax[0].set_title("Comparación de Eigenvalues")
    ax[0].set_xlabel("Parte Real")
    ax[0].set_ylabel("Parte Imaginaria")
    ax[0].axhline(0, color='black', linewidth=0.5)
    ax[0].axvline(0, color='black', linewidth=0.5)
    ax[0].legend()
    
    # Gráfico de los vectores propios
    for i in range(num_functions):
        eigenvectors = eigenvectors_list[i]
        color = colors(i)
        
        for j in range(eigenvectors.shape[1]):
            vector = eigenvectors[:, j]
            
            # Calcular el ángulo del vector con respecto al eje x
            angle = np.angle(vector[0])
            
            # Si el vector apunta hacia la izquierda (ángulo > 90° o < -90°),
            # multiplicarlo por -1 para que apunte hacia la derecha
            if abs(angle) > np.pi/2:
                vector = -vector
                
            # Dibujar la flecha
            ax[1].arrow(0, 0,
                        np.real(vector[0]), np.imag(vector[0]),
                        head_width=0.05, head_length=0.1,
                        fc=color, ec=color, alpha=0.6,
                        length_includes_head=True,
                        label=labels[i] if j == 0 else "")
            
            # Agregar un punto en la punta de la flecha
            ax[1].scatter(np.real(vector[0]), np.imag(vector[0]), 
                          color=color, s=50, marker=f"{i+1}")
    
    ax[1].set_title("Comparación de Eigenvectors")
    ax[1].set_xlabel("Parte Real")
    ax[1].set_ylabel("Parte Imaginaria")
    ax[1].axis('equal')
    ax[1].axhline(0, color='black', linewidth=0.5)
    ax[1].axvline(0, color='black', linewidth=0.5)
    ax[1].legend()
    
    plt.tight_layout()
    plt.show()


def extract_structure(mat_struct):
    matrices = []
    for i in range(mat_struct.size):
        # Acceder a cada elemento de la estructura
        element = mat_struct[0, i]
        matrix_data = {
            'A': element['A'],
            'B': element['B'],
            'eig': {
                'values': element['eig']['values'][0, 0].flatten(),
                'vectors': element['eig']['vectors'][0, 0],
            }
        }
        matrices.append(matrix_data)
    return matrices
 
if __name__ == "__main__":
    case = 9
    A ,B = matrices[case]['A'], matrices[case]['B'] 
    k = len(A)
        
    txt_scipy = 'scipy'
    eigenvalues_scipy, eigenvectors_scipy = eig(A, B)
    
    for j in range(eigenvectors_scipy.shape[1]):
        eigenvectors_scipy[:,j] = eigenvectors_scipy[:,j]/np.linalg.norm(eigenvectors_scipy[:,j])
    
    txt_my = 'my'
    A ,B = convert_to_petsc(A,B)
    eigenvalues_my, eigenvectors_my = eigs(A, B, k=k)
    

    
    data = loadmat('matrices_results.mat')
    
    matrices = data['matrices']
    
    matrices_list = extract_structure(matrices)
    
    txt_matlab = 'matlab'
    eigenvalues_matlab = matrices_list[case-1]['eig']['values']
    eigenvectors_matlab = matrices_list[case-1]['eig']['vectors']

    for j in range(eigenvectors_matlab.shape[1]):
        eigenvectors_matlab[:,j] = eigenvectors_matlab[:,j]/np.linalg.norm(eigenvectors_matlab[:,j])

    eigs_list = [eigenvalues_scipy, eigenvalues_my, eigenvalues_matlab]
    vec_list = [eigenvectors_scipy, eigenvectors_my, eigenvectors_matlab]
    labels = [txt_scipy, txt_my, txt_matlab]    
    
    plot_comparison_eigenvalues_vectors(eigs_list, vec_list, labels)
    