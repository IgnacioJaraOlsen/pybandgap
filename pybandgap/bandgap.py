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
    
    # n_x = matrix_prime.getSize()[0]
    # n_y = matrix_prime.getSize()[1]
    
    # values = matrix_prime.getValues(range(n_x), range(n_y))
    
    # rounded_values = np.round(values, 15)
    # matrix_round = MatExtended().create()
    # matrix_round.setSizes((n_x, n_y))
    # matrix_round.setType(PETSc.Mat.Type.AIJ)
    # matrix_round.setValues(range(n_x), range(n_y), rounded_values)
    # matrix_round.assemblyBegin()
    # matrix_round.assemblyEnd()    
    return matrix_prime

def wave_vector(structure, NINT):
    NINT = int(NINT/3) + 1
    
    # Use structure limits instead of computing them from meshes
    x_min = structure.x_min
    x_max = structure.x_max
    y_min = structure.y_min
    y_max = structure.y_max
    
    Lx = x_max - x_min
    Ly = y_max - y_min
    
    Minv = 1e-4
    X_0_L = np.linspace(Minv / Lx, np.pi / Lx, NINT)
    X_L_0 = np.linspace(np.pi / Lx, Minv / Lx, NINT)
    
    Y_0_L = np.linspace(Minv / Ly, np.pi / Ly, NINT)
    Y_L_0 = np.linspace(np.pi / Ly, Minv / Ly, NINT)

    X_L = np.full(NINT, np.pi / Lx)
    Y_0 = np.full(NINT, Minv / Ly)
    
    thetax = np.hstack((X_0_L[:-1], X_L[:-1], X_L_0))
    thetay = np.hstack((Y_0[:-1], Y_0_L[:-1], Y_L_0))
    
    return thetax, thetay

def eig_bands(structure, mass_matrix, stiffness_matrix, mesh_index=0, NINT=20, N_eig=5,
              tol: float = 1e-10, max_it: int = 200, opt_mode=False):
    
    thetax, thetay = wave_vector(structure, NINT)
    # Create T_matrix for specific mesh from structure
    T_k = T_matrix(structure)
    bands = np.zeros((len(thetax), N_eig))
    
    if opt_mode:
        nn = structure.total_nodes * 2
        phis = np.zeros((len(thetax), N_eig, nn))
    
    for i, (x, y) in enumerate(zip(thetax, thetay)):
        T = T_k(x, y)
        M = set_matrix_prime(mass_matrix, T)
        K = set_matrix_prime(stiffness_matrix, T)
        
        eigenvalues, eigenvectors = solve_generalized_eigenvalue_problem(
            K,
            M,
            nev=N_eig,
            tol=tol,
            max_it=max_it,
        )
        
        bands[i,:] = np.sqrt(np.abs(np.real(eigenvalues[:N_eig])))
        
        if opt_mode:
            phis[i,:,:] = T.matMult(eigenvectors[:N_eig])
    
    if opt_mode:
        return bands, phis

    return bands

def bandgap(n, structure, mass_matrix, stiffness_matrix, mesh_index=0, NINT=20, N_eig=5, 
            plot=True, normalized=1/(2 * np.pi)/1000, tol: float = 1e-10, max_it: int = 200):
    
    bands = eig_bands(
        structure, 
        mass_matrix, 
        stiffness_matrix, 
        mesh_index=mesh_index,
        NINT=NINT, 
        N_eig=N_eig, 
        tol=tol, 
        max_it=max_it
    ) * normalized
    
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

    ax.plot(data, color='blue')
    ax.set_xlim([0, x_lim-1])
    ax.set_ylim([0, np.max(data)])
    
    maximo = np.max(data[:,n-1])
    minimo = np.min(data[:,n])
    delta = minimo - maximo
    media = (maximo + minimo) / 2
    
    txt = r'$\Delta\omega =$ ' + str(round(delta, 2)) + ' [kHz]'
    ax.text((x_lim-1)/2, media, txt, fontsize=12, ha='center', va='center', 
            color='black', weight='bold', style='italic')
    
    ax.fill([0, x_lim, x_lim, 0], [minimo, minimo, maximo, maximo], 'k', 
            linestyle='none', alpha=0.25)
    
    ax.plot([0, x_lim], [maximo, maximo], 'k-', linewidth=0.1)
    ax.plot([0, x_lim], [minimo, minimo], 'k-', linewidth=0.1)
    
    ax.grid(True)
    ax.set_title(rf'n = {n}', fontsize=12)
    ax.set_xlabel(r'Wave vector', fontsize=12)
    ax.set_ylabel(r'Frequency [kHz]', fontsize=12)
    
    ax.set_xticks(np.linspace(0, x_lim-1, 4))
    ax.set_xticklabels([r'$\Gamma$', r'$X_{1}$', r'$M_{1}$', r'$\Gamma$'])
    
    plt.show()