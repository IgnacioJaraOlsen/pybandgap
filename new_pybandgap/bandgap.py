import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from dataclasses import dataclass
from structure import SetStructure
from pybandgap.eigenvalue_solver import solve_generalized_eigenvalue_problem



@dataclass
class BandGap:
    structure: SetStructure
    NINT: int = 100
    n: int = 3
    tol: float = 1e-10
    max_iter: int = 1000
    num_eigs: int = 10
    mode: Literal["simulate", "optimize", "s", 'o'] = "simulate"

    def __post_init__(self):
        self._wave_vector()
        self.bands = np.zeros((len(self.thetax), self.num_eigs))
    
    def _wave_vector(self):
        geometry = self.structure.geometry
        if geometry == "square":
            Lx = self.structure.Lx
            Ly = self.structure.Ly
            
            self.thetax = np.linspace(-np.pi / Lx, np.pi / Lx, self.NINT)
            self.thetay = np.linspace(-np.pi / Ly, np.pi / Ly, self.NINT)
            
            self.wave_symbols = [r'$\Gamma$', r'$X_{1}$', r'$M_{1}$', r'$\Gamma$']
            

    
    def eig_bands(self):
        for i, (x, y) in enumerate(zip(self.thetax, self.thetay)):
            T_k = self.structure.T_matrix(x, y)
            matrix_prime = set_matrix_prime(self.structure, T_k)
                
            eigvals, eigvecs = solve_generalized_eigenvalue_problem(matrix_prime, self.structure, self.tol, self.max_iter, self.num_eigs)
            self.bands[i, :] = np.sort(np.sqrt(eigvals))
            
            if self.mode == "optimize" or self.mode == "o":
                self.phis[i, :, :] = T_k.matMult(eigvecs)
    
    def plot_bandgap(self):
        
        data = self.bands
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
        
        ax.set_xticks(np.linspace(0, x_lim-1, len(self.wave_symbols)))
        ax.set_xticklabels(self.wave_symbols)
        
        plt.show()        