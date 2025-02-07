import nlopt
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc
from typing import Literal
from dataclasses import dataclass
from pybandgap_v2.structure import SetStructure
from pybandgap_v2.eigs import calc_eigs

@dataclass
class BandGap:
    structure: SetStructure
    NINT: int = 60
    n: int = 3
    tol: float = 1e-20
    max_iter: int = 1000
    num_eigs: int = 6
    mode: Literal["simulate", "optimize", "s", "0"] = "simulate"
    parameters_optimize = None

    def __post_init__(self):
        self._wave_vector()
        self.bands = np.zeros((len(self.thetax), self.num_eigs))
        if self.mode in ("optimize", "o"):
            columns = self.structure.total_nodes * 2
            self.values = np.zeros((len(self.thetax), self.num_eigs), dtype=np.complex128)
            self.phis = np.zeros((len(self.thetax), self.num_eigs, columns), dtype=np.complex128)
    
    def _wave_vector(self):

        IBZ_points = self.structure.IBZ_points
        
        points = []
        
        for i in range(len(IBZ_points) - 1):
            p1 = IBZ_points[i]
            p2 = IBZ_points[(i + 1) % len(IBZ_points)]
            points.append(p1)
            intermediate_points = np.linspace(p1, p2, int(self.NINT / len(IBZ_points)) + 2)[1:-1]
            points.extend(intermediate_points)
        
        points = np.vstack((np.array(points), IBZ_points[-1])) - self.structure.mid_point
        
        points = points.T
        
        self.thetax = points[0,:] * 2* np.pi / (2 * np.max(points[0,:]))**2 + 0.00001
        self.thetay = points[1,:] * 2* np.pi / (2 * np.max(points[1,:]))**2 + 0.00001
                   
    def set_matrix_prime(self, T):
        T_T = PETSc.Mat()
        M_prime = PETSc.Mat()
        K_prime = PETSc.Mat()
        
        T.hermitianTranspose(out=T_T)
        
        T_T.matMatMult(self.structure.M, T, result = M_prime)
        T_T.matMatMult(self.structure.K, T, result = K_prime)
        
        return M_prime, K_prime
    
    @staticmethod
    def round_matrix_values(matrix, decimals=15):
        size = matrix.getSize()  # Tamaño de la matriz
        rows, cols = matrix.getOwnershipRange()  # Rango de índices propios
        
        # Crear una nueva matriz para los valores redondeados
        rounded_matrix = PETSc.Mat().create()
        rounded_matrix.setSizes(size)
        rounded_matrix.setFromOptions()
        rounded_matrix.setUp()
        
        # Iterar sobre los valores de la matriz
        for i in range(rows, cols):
            cols_indices, values = matrix.getRow(i)  # Obtener fila actual
            rounded_values = [round(v, decimals) for v in values]  # Redondear valores
            rounded_matrix.setValues(i, cols_indices, rounded_values)
        
        rounded_matrix.assemble()  # Ensamblar la nueva matriz
        return rounded_matrix
    
    def eig_bands(self):
        self.structure.Mass_and_Stiffness_matrix()
        for i, (x, y) in enumerate(zip(self.thetax, self.thetay)):
            self.structure.T_matrix(x, y)
            T_k = self.structure.T
            
            M_prime, K_prime = self.set_matrix_prime(T_k)
            
            M_prime, K_prime = self.round_matrix_values(M_prime), self.round_matrix_values(K_prime)
                
            values, vectors = calc_eigs(K_prime, M_prime, n_eigs = self.num_eigs, which = 'SM')
            
            self.bands[i, :] = np.sqrt(np.abs(np.real(values)))
            
            if self.mode == "optimize" or self.mode == "o":
                self.values[i, :] = values
                T_k = np.array(T_k[:,:])
                for j, eigvec in enumerate(vectors.T):
                    transformed_vec = T_k @ eigvec.reshape(-1, 1)
                    M = self.structure.M[:,:]
                    norm_factor = np.sqrt((transformed_vec.T @ M @ transformed_vec).item())
                    normalized_vec = (transformed_vec / norm_factor).reshape(1, -1)
                    self.phis[i, j, :] = normalized_vec

    def plot_bandgap(self):
        
        self.eig_bands()
        
        data = self.bands/(2 * np.pi)/1000
        n = self.n
        
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

    @staticmethod
    def compute_eigenvalue_derivative(dK_dx, dM_dx, eigenvector, eigenvalue, M):

        evec_petsc = dK_dx.createVecLeft()
        temp_vec1 = dK_dx.createVecLeft()
        temp_vec2 = dK_dx.createVecLeft()
        
        evec_petsc.array[:] = eigenvector
        
        dK_dx.mult(evec_petsc, temp_vec1)
          
        dM_dx.mult(evec_petsc, temp_vec2)
        
        temp_vec2.scale(eigenvalue**2)
        
        temp_vec1.axpy(-1.0, temp_vec2)

        evec_conj = evec_petsc.copy()
        evec_conj.conjugate()

        numerator = evec_conj.dot(temp_vec1)
        
        M.mult(evec_petsc, temp_vec2)
        
        denominator = 2 * np.sqrt(np.abs(np.real(eigenvalue)))
        
        return numerator / denominator
                
    def objective(self, P):
        self.eig_bands()
        if P == 0:
            data = self.bands
            n = self.n
            max_w_n = np.max(data[:,n-1])
            min_w_n_p_1 = np.min(data[:,n])
            delta = min_w_n_p_1 - max_w_n
            return delta
    
    def d_objective(self, P):
        if P == 0:
            data = self.bands
            n = self.n
            max_idx = np.argmax(data[:,n-1])
            min_idx = np.argmin(data[:,n])

            def d_w(i,j, dK, dM):
                
                u = self.phis[i,j,:]
                w = self.values[i,j]
                return self.compute_eigenvalue_derivative(dK, dM, u, w, self.structure.M)
        
            for fem_idx, fem in enumerate(self.structure.fems):
                S  = fem.S
                for parameter in fem.parameters:
                    derivatives = np.zeros(fem.N_elements)
                    dw_n = np.zeros(fem.N_elements,dtype=np.complex128) 
                    dw_n_1 = np.zeros(fem.N_elements,dtype=np.complex128) 
                    for element in range(fem.N_elements):
                        dM, dK = self.structure.Mass_and_Stiffness_derivate(element, parameter, fem_idx)
                        dw_n[element] =  d_w(max_idx, n-1, dK, dM)
                        dw_n_1[element] =  d_w(min_idx, n, dK, dM)
                    
                    derivatives = np.real(dw_n_1 - dw_n)
                    index = self.structure.x_map[fem_idx][parameter]
                    values = np.dot(S.T, derivatives.reshape(-1, 1))
                    self.structure.d_x[index] = values.reshape(1, -1)
                    
    def optimizer(self):
        def myfunc(x, grad):
            self.structure.x = x
            self.structure.apply_x()
            
            obj = -self.objective(0)
            
            if grad.size > 0:
                self.d_objective(0)
                grad[:] = -self.structure.d_x
                
            return obj
        
        opt = nlopt.opt(nlopt.LD_MMA, len(self.structure.x))

        opt.set_lower_bounds(np.zeros_like(self.structure.x))
        opt.set_upper_bounds(np.ones_like(self.structure.x))
        opt.set_min_objective(myfunc)
        opt.set_ftol_rel(1e-5)
        x = np.round(opt.optimize(self.structure.x))
        self.structure.x = x
        self.structure.apply_x()
        self.plot_bandgap()
        
    def test(self):
        x_m = np.zeros(12)
        x_A = np.zeros(12)

        mask_x_m = np.where(np.isin(self.structure.IBZ_elements[0], np.array([1, 2, 4, 8])))[0]
        mask_x_A = np.where(np.isin(self.structure.IBZ_elements[0], np.array([1, 2, 4, 8, 13])))[0]

        x_m[mask_x_m] = 1
        x_A[mask_x_A] = 1

        self.structure.x[self.structure.x_map[0]['x_m']] = x_m
        self.structure.x[self.structure.x_map[0]['x_A']] = x_A
        
        self.structure.apply_x()
        
        self.objective(0)

        self.d_objective(0)
        
        print(self.structure.d_x.reshape(-1,1))
        
        self.plot_bandgap()

    def test2(self):
        x_m = np.zeros(12)
        x_A = np.zeros(12)

        mask_x_m = np.where(np.isin(self.structure.IBZ_elements[0], np.array([1, 2, 4, 8])))[0]
        mask_x_A = np.where(np.isin(self.structure.IBZ_elements[0], np.array([1, 2, 4, 8, 13])))[0]

        x_m[mask_x_m] = 1
        x_A[mask_x_A] = 1

        self.structure.x[self.structure.x_map[0]['x_m']] = x_m
        self.structure.x[self.structure.x_map[0]['x_A']] = x_A
        
        self.structure.apply_x()
        
        self.structure.Mass_and_Stiffness_matrix()
        
        self.structure.T_matrix(0, 0)
        
        M_prime, K_prime = self.set_matrix_prime(self.structure.T)
        
        dM, dK = self.structure.Mass_and_Stiffness_derivate(0, 'x_m', 0)
        
        from slepc_eigs import calc_eigs
        from sensitivity_eig import compute_eigen_derivative
        from tabulate import tabulate
        
        values, vectors = calc_eigs(K_prime, M_prime, n_eigs = 6, which = 'SM')
        
        vectors = self.structure.T[:,:] @ vectors
        
        table_data = []
        
        for i in range(6):
            mu = values[i]
            u = vectors[:,i]
            deriv = compute_eigen_derivative(mu, u, dK, dM, self.structure.M, backend='petsc')
            table_data.append([i, mu, deriv])
            
        headers = ['Index', 'Value', 'Derivative']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        