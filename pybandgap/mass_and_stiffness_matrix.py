import numpy as np
import dolfinx.fem.petsc
import ufl
from mpi4py import MPI

class Material:
    # Variable de clase para contar el número de instancias
    counter = 0

    def __init__(self, material, young_modulus, poisson_ratio, density):
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.density = density
        self.material = material

        # Incrementar el contador cada vez que se crea una nueva instancia
        Material.counter += 1
        # Asignar el número de creación a la instancia
        self.creation_number = Material.counter

    def __repr__(self):
        return f'{self.material} (Creation Number: {self.creation_number})'

    @classmethod
    def get_counter(cls):
        return cls.counter


def multiple_materials_matrix(mesh, materials_map, diameter = None):
    V0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    
    young_moduli = np.zeros(mesh.topology.index_map(mesh.topology.dim).size_local)
    poisson_ratios = np.zeros_like(young_moduli)
    densities = np.zeros_like(young_moduli)

    # Llenar arrays con propiedades de cada material
    for cell_index, material in materials_map.items():
        young_moduli[cell_index] = material.young_modulus
        poisson_ratios[cell_index] = material.poisson_ratio
        densities[cell_index] = material.density
        
    # Funciones para propiedades materiales
    E_func = dolfinx.fem.Function(V0)
    E_func.x.array[:] = young_moduli

    nu_func = dolfinx.fem.Function(V0)
    nu_func.x.array[:] = poisson_ratios

    density_func = dolfinx.fem.Function(V0)
    density_func.x.array[:] = densities

    V = dolfinx.fem.functionspace(mesh, ("CG", 1, (mesh.topology.dim,)))
    

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    mass_form = density_func * ufl.inner(u, v) * ufl.dx
    mass_matrix = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(mass_form))
    mass_matrix.assemble()

    # Forma de la matriz de rigidez con dos grados de libertad
    def stress_strain(u, v, E, nu):
        # Matriz de elasticidad 2D
        epsilon = ufl.sym(ufl.grad(u))
        sigma = E / (1 + nu) * (epsilon + nu/(1-nu) * ufl.tr(epsilon) * ufl.Identity(2))
        return ufl.inner(sigma, ufl.grad(v)) * ufl.dx
    
    
    
    stiffness_form = stress_strain(u, v, E_func, nu_func)
    stiffness_matrix = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(stiffness_form))
    stiffness_matrix.assemble()
    
    return mass_matrix, stiffness_matrix


def new_multiple_materials_matrix(mesh, materials_map, radii_map=None):
    # Verificación de que se proporcionen los radios para cada elemento
    if mesh.topology.dim == 1 and radii_map is None:
        raise ValueError("Se deben definir los radios de los elementos de la estructura de enrejado (radii_map)")
  
    # Espacio de función para propiedades constantes por elemento
    V0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    
    # Inicialización de arrays para propiedades de materiales
    young_moduli = np.zeros(mesh.topology.index_map(mesh.topology.dim).size_local)
    poisson_ratios = np.zeros_like(young_moduli)
    densities = np.zeros_like(young_moduli)
    cross_sectional_areas = np.zeros_like(young_moduli)

    # Llenar arrays con propiedades de cada material
    for cell_index, material in materials_map.items():
        young_moduli[cell_index] = material.young_modulus
        poisson_ratios[cell_index] = material.poisson_ratio
        densities[cell_index] = material.density
        
        # Calcular área de la sección transversal para cada barra
        cross_sectional_areas[cell_index] = np.pi * radii_map[cell_index]**2 /4
        
    # Funciones para propiedades materiales
    E_func = dolfinx.fem.Function(V0)
    E_func.x.array[:] = young_moduli

    nu_func = dolfinx.fem.Function(V0)
    nu_func.x.array[:] = poisson_ratios

    density_func = dolfinx.fem.Function(V0)
    density_func.x.array[:] = densities

    # Área de sección transversal como función
    area_func = dolfinx.fem.Function(V0)
    area_func.x.array[:] = cross_sectional_areas
    
    # Espacio de función para desplazamientos
    V = dolfinx.fem.functionspace(mesh, ("CG", 1, (2,)))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Forma de la matriz de masa considerando el área de la sección transversal
    mass_form = density_func * area_func * ufl.inner(u, v) * ufl.dx
    mass_matrix = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(mass_form))
    mass_matrix.assemble()

    # Forma de la matriz de rigidez para estructuras de enrejado (barras)
    def truss_stiffness(u, v, E, area):
        # Para estructuras de enrejado (1D), la deformación es axial
        # Strain = du/dx (derivada del desplazamiento en la dirección del elemento)
        # Stress = E * Strain
        # Stiffness considera el área de la sección transversal
        return E * area * ufl.inner(ufl.tr(ufl.grad(u))*ufl.Identity(2), ufl.sym(ufl.grad(v))) * ufl.dx
    
    # Forma de rigidez usando la función de rigidez de enrejado
    stiffness_form = truss_stiffness(u, v, E_func, area_func)
    stiffness_matrix = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(stiffness_form))
    stiffness_matrix.assemble()
    
    return mass_matrix, stiffness_matrix