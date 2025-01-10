import nlopt
import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc
from typing import Literal
from dataclasses import dataclass
from pybandgap.structure import SetStructure
from eigenvalue_solver import solve_generalized_eigenvalue_problem

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
        
        self.thetax = points[0,:] * 2* np.pi / (2 * np.max(points[0,:]))**2
        self.thetay = points[1,:] * 2* np.pi / (2 * np.max(points[1,:]))**2
                   
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
                
            eigvals, eigvecs = solve_generalized_eigenvalue_problem(K_prime, M_prime, tol= self.tol, max_it=self.max_iter, nev=self.num_eigs)
            
            self.bands[i, :] = np.sqrt(np.abs(np.real(eigvals)))
            
            if self.mode == "optimize" or self.mode == "o":
                T_k = np.array(T_k[:,:])
                for j, eigvec in enumerate(eigvecs):
                    self.phis[i, j, :] = (T_k@eigvec.reshape(-1,1)).reshape(1,-1)

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

        temp_vec1 = eigenvector.duplicate()
        temp_vec2 = eigenvector.duplicate()

        # dK_dx * u
        dK_dx.mult(eigenvector, temp_vec1)

        # dM_dx * u
        dM_dx.mult(eigenvector, temp_vec2)
        
        #u^T * (dK_dx - ω^2 * dM_dx) * u
        numerator1 = eigenvector.dot(temp_vec1) # u^T * (dK_dx * u)
        numerator2 = eigenvalue**2 * eigenvector.dot(temp_vec2) # ω^2 * u^T * (dM_dx * u)
        numerator = numerator1 - numerator2

        # Calcular denominador: 2 * ω * u^T * M * u
        temp_vec3 = temp_vec2.duplicate()
        M.mult(eigenvector, temp_vec3) # M * u
        denominator = 2 * eigenvalue * eigenvector.dot(temp_vec3) # 2 * ω * u^T * temp_vec3
        
        # Calcular la derivada final
        derivative = numerator / denominator
          
        return derivative
                
    def objective(self, P):
        self.eig_bands()
        if P == 0:
            data = self.bands/(2 * np.pi)/1000
            n = self.n
            max_w_n = np.max(data[:,n-1])
            min_w_n_p_1 = np.min(data[:,n])
            delta = min_w_n_p_1 - max_w_n
            return delta
    
    def d_objective(self, P):
        if P == 0:
            data = self.bands/(2 * np.pi)/1000
            n = self.n
            max_idx = np.argmax(data[:,n-1])
            min_idx = np.argmin(data[:,n])

            def d_w(i,j, dK, dM):
                
                u = self.phis[i,j,:]
                u = PETSc.Vec().createWithArray(u)
                w = self.bands[i,j]
                return self.compute_eigenvalue_derivative(dK, dM, u, w, self.structure.M)
        
            for fem_idx, fem in enumerate(self.structure.fems):
                S  = fem.S
                for parameter in fem.parameters:
                    derivatives = np.zeros(fem.N_elements)
                    dw_n = np.zeros(fem.N_elements) 
                    dw_n_1 = np.zeros(fem.N_elements) 
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
        opt.set_maxeval(50)
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
        
        self.plot_bandgap()
        for i in np.sort(self.structure.d_x):
            print(f'{i:.3e}')
        