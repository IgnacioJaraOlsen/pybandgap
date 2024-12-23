import numpy as np
from petsc4py import PETSc
import matplotlib.pyplot as plt
from pybandgap.T_matrix import T_matrix, MatExtended
from pybandgap.eigenvalue_solver import solve_generalized_eigenvalue_problem

def set_matrix_prime(matrix, T):
    T = T.copy()
    T_T = T.copy().conjugate_transpose()
    M = matrix.copy()
    matrix_prime = T_T.matMatMult(M, T)
    
    n_x = matrix_prime.getSize()[0]
    n_y = matrix_prime.getSize()[1]
    
    values  = matrix_prime.getValues(range(n_x), range(n_y))
    
    rounded_values = np.round(values, 15)
    matrix_round = MatExtended().create()
    matrix_round.setSizes((n_x, n_y))
    matrix_round.setType(PETSc.Mat.Type.AIJ)
    matrix_round.setValues(range(n_x), range(n_y), rounded_values)
    matrix_round.assemblyBegin()
    matrix_round.assemblyEnd()    
    return matrix_round

def wave_vector(meshes, NINT):
    
    NINT = int(NINT/3) + 1
    
    x_min = float('inf')
    x_max = float('-inf')
    y_min = float('inf')
    y_max = float('-inf')
    
    if not isinstance(meshes, (list, tuple)):
        meshes = [meshes]
    
    for mesh in meshes:
        x = mesh.geometry.x
        x_min = min(x_min, np.min(x[:, 0]))
        x_max = max(x_max, np.max(x[:, 0]))
        y_min = min(y_min, np.min(x[:, 1]))
        y_max = max(y_max, np.max(x[:, 1]))
    
    Lx = x_max - x_min
    Ly = y_max - y_min
    
    Minv = 1e-4
    X_0_L = np.linspace(Minv / Lx, np.pi / Lx, NINT)
    X_L_0 = np.linspace(np.pi / Lx, Minv / Lx, NINT)
    
    Y_0_L = np.linspace(Minv / Ly, np.pi / Ly, NINT)
    Y_L_0 = np.linspace(np.pi / Ly, Minv / Ly, NINT)

    X_L = np.full(NINT, np.pi / Lx)
    Y_0 = np.full(NINT, Minv / Ly)
    
    thetax = np.hstack((X_0_L[:-1], X_L[:-1], X_L_0[:-1]))
    thetay = np.hstack((Y_0[:-1], Y_0_L[:-1], Y_L_0[:-1]))
    
    return thetax, thetay


def eig_bands(mesh, mass_matrix, stiffness_matrix, NINT = 20, N_eig= 5,
              tol: float = 1e-10,
              max_it: int = 200,):

    thetax, thetay = wave_vector(mesh, NINT)
    T_k = T_matrix(mesh)
    bands = np.zeros((len(thetax), N_eig))
       
    for i, (x, y) in enumerate(zip(thetax, thetay)):
        T = T_k(x, y)
        M = set_matrix_prime(mass_matrix, T)
        K = set_matrix_prime(stiffness_matrix, T)

        eigensolver = solve_generalized_eigenvalue_problem(
            K,
            M,
            nev=N_eig,
            tol= tol,
            max_it= max_it,
            )
        
        bands[i,:] = np.sqrt(eigensolver[:N_eig])

    return bands
    
def bandgap(n, mesh, mass_matrix, stiffness_matrix, NINT = 20, N_eig= 5, plot = True,
            normalized = 1/(2 * np.pi)/1000,
            tol: float = 1e-10,
            max_it: int = 200,):
    
    bands = eig_bands(mesh, mass_matrix, stiffness_matrix, NINT = NINT, N_eig= N_eig, tol = tol, max_it= max_it) * normalized
    
    maximo = np.max(bands[:, n - 1])
    minimo = np.min(bands[:, n])
    delta = minimo - maximo
    medium_frequency = (minimo + maximo)/2
    
    if plot:
        plot_bands(bands, n)
    
    return delta, medium_frequency, bands


def plot_bands(data, n):
    fig, ax = plt.subplots()
    
    x_lim = data.shape[0]

    # Usamos el argumento `color` para especificar el color azul
    ax.plot(data, color='blue')
    
    # # Establecemos los límites de los ejes
    ax.set_xlim([0, x_lim-1])  # Límites del eje x
    ax.set_ylim([0, np.max(data)])  # Límites del eje y
    
    # Calcular maximo, minimo, delta y media para la zona rellena
    maximo = np.max(data[:,n-1])  # Maximo de la columna n (Python usa índices 0-basados)
    minimo = np.min(data[:,n])    # Minimo de la columna n+1
    delta = minimo - maximo
    media = (maximo + minimo) / 2
    
    # Añadir texto en la gráfica
    txt = r'$\Delta\omega =$ ' + str(round(delta, 2)) + ' [kHz]'
    ax.text((x_lim-1)/2, media, txt, fontsize=12, ha='center', va='center', 
            color='black', weight='bold', style='italic')
    
    # # Rellenar el área entre maximo y minimo
    ax.fill([0, x_lim, x_lim, 0], [minimo, minimo, maximo, maximo], 'k', 
            linestyle='none', alpha=0.25)
    
    # Añadir líneas horizontales en maximo y minimo
    ax.plot([0, x_lim], [maximo, maximo], 'k-', linewidth=0.1)
    ax.plot([0, x_lim], [minimo, minimo], 'k-', linewidth=0.1)
    
    # # Añadir la grilla
    ax.grid(True)
    
    # # Título de la gráfica con LaTeX habilitado
    ax.set_title(rf'n = {n}', fontsize=12)
    
    # Etiquetas de los ejes con formato LaTeX
    ax.set_xlabel(r'Wave vector', fontsize=12)
    ax.set_ylabel(r'Frequency [kHz]', fontsize=12)
    
    # Establecemos las posiciones de las etiquetas en el eje x\
    ax.set_xticks(np.linspace(0, x_lim-1, 4))  # Posiciones específicas en el eje x
    
    # Asignamos las etiquetas del eje x
    ax.set_xticklabels([r'$\Gamma$', r'$X_{1}$', r'$M_{1}$', r'$\Gamma$'])
    
    plt.show()
    

if __name__ == "__main__":
    from dolfinx import mesh
    from mpi4py import MPI
    n = 2
    msh = mesh.create_unit_square(MPI.COMM_WORLD,n,n)
    T = T_matrix(msh)
    T = T(30,30)
    n_x = T.getSize()[0]
    n_y = T.getSize()[1]
    print(n_x, n_y)
    print(T.getValues(range(n_x), range(n_y)))
    
    
    