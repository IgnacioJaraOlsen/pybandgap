import numpy as np
from petsc4py import PETSc
import matplotlib.pyplot as plt
from pybandgap.eigenvalue_solver import solve_generalized_eigenvalue_problem

# This class extends PETSc.Mat to add conjugate_transpose (Hermitian transpose)
class MatExtended(PETSc.Mat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def conjugate_transpose(self):
        # Get the transpose of the matrix
        mat_transpose = self.transpose()
        
        # Apply conjugation to the transpose
        mat_conjugate_transpose = mat_transpose.conjugate()
        
        return mat_conjugate_transpose

# Create a transformation matrix using PETSc for mesh operations
def T_matrix(mesh):
    x = mesh.geometry.x
    x_min = np.min(x[:, 0])
    x_max = np.max(x[:, 0])
    y_min = np.min(x[:, 1])
    y_max = np.max(x[:, 1])  
    
    #Find node indices that satisfy a given condition
    def get_node_indices(condition):
        return np.where(np.apply_along_axis(condition, 1, x))[0]
    
    corner_down_left = get_node_indices(lambda coord: np.allclose(coord, [x_min, y_min, 0]))
    
    corner_down_rigth = get_node_indices(lambda coord: np.allclose(coord, [x_max, y_min, 0]))
    
    corner_up_rigth = get_node_indices(lambda coord: np.allclose(coord, [x_max, y_max, 0]))
    
    corner_up_left = get_node_indices(lambda coord: np.allclose(coord, [x_min, y_max, 0]))
    
    corner_indices = np.sort(np.hstack((corner_down_rigth, corner_up_rigth, corner_up_left)))
    
    def line_condition(coord, axis, t_d_l_r):
        line =  np.array([[x_min, x_max],[y_min,y_max]])
        in_line = np.isclose(coord[1 - axis], line[1 - axis, t_d_l_r])
        less_max = coord[axis] < line[axis][1]
        over_min = coord[axis] > line[axis][0]
        return np.all([in_line, less_max, over_min])
    
    bottom_indices = get_node_indices(lambda coord: line_condition(coord, 0, 0))
    
    top_indices = get_node_indices(lambda coord: line_condition(coord, 0, 1))

    left_indices = get_node_indices(lambda coord: line_condition(coord, 1, 0))
    
    right_indices = get_node_indices(lambda coord: line_condition(coord, 1, 1))
    
    # Create transformation matrix using PETSc
    num_nodes_x= x.shape[0]
    less_nodes = np.sort(np.hstack((corner_indices, top_indices, right_indices)))
    
    all_nodes = np.arange(num_nodes_x)
    
    reduced_nodes = np.setdiff1d(all_nodes, less_nodes)
    num_nodes_y = len(reduced_nodes)
    
    # Create PETSc matrix
    T = MatExtended().create()
    T.setSizes((2 * num_nodes_x, 2 * (num_nodes_y)))
    T.setType(PETSc.Mat.Type.AIJ)
    T.setUp()
    
    # Set matrix values
    def set_matrix_values():
        # Corner nodes relationship
        for i in corner_indices:
            column = np.where(reduced_nodes == corner_down_left)[0]
            T.setValue(2*i, 2*column, 1.0)    # x displacement
            T.setValue(2*i+1, 2*column+1, 1.0)  # y displacement

        # # # Top and bottom edges relationship
        for i, j in zip(top_indices, bottom_indices):
            column = np.where(reduced_nodes == j)[0]
            T.setValue(2*i, 2*column, 1.0)       # x displacement
            T.setValue(2*i+1, 2*column+1, 1.0)   # y displacement

        # # # Right and left edges relationship
        for i, j in zip(right_indices, left_indices):
            column = np.where(reduced_nodes == j)[0]
            T.setValue(2*i, 2*column, 1.0)       # x displacement
            T.setValue(2*i+1, 2*column+1, 1.0)   # y displacement

        # Ensure diagonal 1s for unconstrained nodes
        
        for i in range(2 * num_nodes_x):
            if i in reduced_nodes:
                column = np.where(reduced_nodes == i)[0]
                T.setValue(2*i, 2*column, 1.0)
                T.setValue(2*i+1, 2*column+1, 1.0)

        T.assemblyBegin()
        T.assemblyEnd()
        return T

    # Create the transformation matrix
    T_matrix = set_matrix_values()

    # Generate a wave vector-dependent transformation matrix
    def T_matrix_k(k_x, k_y):
        Lx = x_max - x_min
        Ly = y_max - y_min
        # rigth nodes
        
        def change_values(index_row, index_columns, value):
            columns = np.where(np.isin(reduced_nodes, index_columns))[0].astype(np.int32)
            rows = index_row.astype(np.int32)
            for i, j in zip(rows, columns):
                T_matrix.setValue(2*i, 2*j, value)
                T_matrix.setValue(2*i+1, 2*j+1, value)

        change_values(right_indices, left_indices, np.exp(1j*Lx*k_x))
        change_values(top_indices, bottom_indices, np.exp(1j*Ly*k_y))
        change_values(corner_up_left, corner_down_left, np.exp(1j*Ly*k_y))
        change_values(corner_down_rigth, corner_down_left, np.exp(1j*Lx*k_x))
        change_values(corner_up_rigth, corner_down_left, np.exp(1j*(k_x* Lx + k_y * Ly)))
        
        T_matrix.assemblyBegin()
        T_matrix.assemblyEnd()
        
        return T_matrix

    return T_matrix_k


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

def wave_vector(mesh, NINT):
    x = mesh.geometry.x
    x_min = np.min(x[:, 0])
    x_max = np.max(x[:, 0])
    y_min = np.min(x[:, 1])
    y_max = np.max(x[:, 1])
    
    Lx = x_max - x_min
    Ly = y_max - y_min
    
    Minv = 1e-4
    X_0_L = np.linspace(Minv / Lx, np.pi / Lx, NINT)
    X_L_0 = np.linspace(np.pi / Lx, Minv / Lx, NINT)
    
    Y_0_L = np.linspace(Minv / Ly, np.pi / Ly, NINT)
    Y_L_0 = np.linspace(np.pi / Ly, Minv / Ly, NINT)

    X_L = np.full(NINT, np.pi / Lx)
    Y_0 = np.full(NINT, Minv / Ly)
    
    thetax = np.hstack((X_0_L, X_L, X_L_0))
    thetay = np.hstack((Y_0, Y_0_L, Y_L_0))
    
    return thetax, thetay


def eig_bands(mesh, mass_matrix, stiffness_matrix, NINT = 20, N_eig= 5):

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
            )
        
        bands[i,:] = np.sqrt(eigensolver[:N_eig])

    return bands
    
def bandgap(n, mesh, mass_matrix, stiffness_matrix, NINT = 20, N_eig= 5, plot = True):
    
    bands = eig_bands(mesh, mass_matrix, stiffness_matrix, NINT = NINT, N_eig= N_eig)/(2 * np.pi)/1000
    
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
    
    
    