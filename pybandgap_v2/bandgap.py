import numpy as np
import matplotlib.pyplot as plt
from petsc4py import PETSc
from typing import Literal
from dataclasses import dataclass
from pybandgap.structure import SetStructure
from pybandgap.eigenvalue_solver import solve_generalized_eigenvalue_problem



@dataclass
class BandGap:
    structure: SetStructure
    NINT: int = 60
    n: int = 3
    tol: float = 1e-10
    max_iter: int = 1000
    num_eigs: int = 6
    mode: Literal["simulate", "optimize", "s", "0"] = "simulate"
    parameters_optimize = None

    def __post_init__(self):
        self._wave_vector()
        self.bands = np.zeros((len(self.thetax), self.num_eigs))
        if self.mode in ("simulate", "o"):
            columns = self.structure.total_nodes * 2
            self.phis = np.zeros((len(self.thetax), self.num_eigs, columns))
    
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
    
    def eig_bands(self):
        self.structure.Mass_and_Stiffness_matrix()
        for i, (x, y) in enumerate(zip(self.thetax, self.thetay)):
            self.structure.T_matrix(x, y)
            T_k = self.structure.T
            
            M_prime, K_prime = self.set_matrix_prime(T_k)
                
            eigvals, eigvecs = solve_generalized_eigenvalue_problem(K_prime, M_prime, tol= self.tol, max_it=self.max_iter, nev=self.num_eigs)
            self.bands[i, :] = np.sqrt(np.abs(np.real(eigvals[:self.num_eigs])))
            
            if self.mode == "optimize" or self.mode == "o":
                self.phis[i, :] = np.array([T_k*eigvec for eigvec in eigvecs[:self.num_eigs]])
    
    
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

    def get_d_w(self, element, prop, variable):
        d_w = np.array((len(self.thetax), self.num_eigs))
        dM, dK = self.structure.Mass_and_Stiffness_derivate(element, prop, variable)
        for i in range(d_w.shape[0]):
            for j in range(d_w.shape[1]):
                u = self.phis[i,j,:]
                u = PETSc.Vec().createWithArray(u)
                w = self.bands[i,j]
                d_w[i,j] = self.compute_eigenvalue_derivative(dM, dK, u, w)
                
    def objective(self, P):
        pass
    
    def d_objective(self, P, ):
        pass