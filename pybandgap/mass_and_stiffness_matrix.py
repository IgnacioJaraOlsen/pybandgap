import numpy as np
import dolfinx.fem.petsc
import ufl

def truss_stiffness(u, v, E, area):
    return E * area * ufl.inner(ufl.tr(ufl.grad(u))*ufl.Identity(2), ufl.sym(ufl.grad(v))) * ufl.dx

def truss_mass(u, v, density, area_func):
    return density * area_func * ufl.inner(u, v) * ufl.dx

def plate_stiffness(u, v, E, nu):
    mu = E / (2 * (1 + nu))
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
        
    epsilon = lambda u: ufl.sym(ufl.grad(u))
    sigma = lambda u: lambda_ * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2.0 * mu * epsilon(u)
        
    return ufl.inner(sigma(u), epsilon(v)) * ufl.dx

def plate_mass(u, v, density):
    return  density * ufl.inner(u, v) * ufl.dx

def matrix_and_stiffness_matrix(mesh, props):
    V0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    V = dolfinx.fem.functionspace(mesh, ("CG", 1, (2,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    E_func = dolfinx.fem.Function(V0)
    E_func.x.array[:] = [material.young_modulus for material in props.materials.values()]
    
    density_func = dolfinx.fem.Function(V0)
    density_func.x.array[:] = [ material.density for material in props.materials.values()]
    
    if mesh.topology.dim == 1:
        if not hasattr(props, 'diameters'):
            raise ValueError("add diameters_map in props")
        area_func = dolfinx.fem.Function(V0)
        area_func.x.array[:] = [np.pi * diameter**2 /4 for diameter in props.diameters.values()]
        
        stiffness_form = truss_stiffness(u, v, E_func, area_func)
        mass_form = truss_mass(u, v, density_func, area_func)
        
    else:
        nu_func = dolfinx.fem.Function(V0)
        nu_func.x.array[:] = [material.poisson_ratio for material in props.materials.values()]

        stiffness_form = plate_stiffness(u, v, E_func, nu_func)
        mass_form = plate_mass(u, v, density_func)

    mass_matrix = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(mass_form))
    mass_matrix.assemble()
    
    stiffness_matrix = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(stiffness_form))
    stiffness_matrix.assemble()
    
    return mass_matrix, stiffness_matrix

def multiple_materials_matrix(mesh, materials_map, diameters_map=None):
    
    young_moduli = np.zeros(mesh.topology.index_map(mesh.topology.dim).size_local)
    poisson_ratios = np.zeros_like(young_moduli)
    densities = np.zeros_like(young_moduli)
    
    V0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    
    if mesh.topology.dim == 1:
        if diameters_map is None:
            raise ValueError("Se deben definir los radios de los elementos de la estructura de enrejado (diameters_map)")

        cross_sectional_areas = np.zeros_like(young_moduli)
        
        for cell_index, material in materials_map.items():
             young_moduli[cell_index] = material.young_modulus
             cross_sectional_areas[cell_index] = np.pi * diameters_map[cell_index]**2 /4
             densities[cell_index] = material.density

        area_func = dolfinx.fem.Function(V0)
        area_func.x.array[:] = cross_sectional_areas
  
    else:
        for cell_index, material in materials_map.items():
            young_moduli[cell_index] = material.young_modulus
            poisson_ratios[cell_index] = material.poisson_ratio
            densities[cell_index] = material.density
        
        nu_func = dolfinx.fem.Function(V0)
        nu_func.x.array[:] = poisson_ratios
        
    E_func = dolfinx.fem.Function(V0)
    E_func.x.array[:] = young_moduli

    density_func = dolfinx.fem.Function(V0)
    density_func.x.array[:] = densities

    V = dolfinx.fem.functionspace(mesh, ("CG", 1, (2,)))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    mass_form = density_func * area_func * ufl.inner(u, v) * ufl.dx

    def truss_stiffness(u, v, E, area):
        return E * area * ufl.inner(ufl.tr(ufl.grad(u))*ufl.Identity(2), ufl.sym(ufl.grad(v))) * ufl.dx
    
    def plate_stiffness(u, v, E, nu):
        mu = E / (2 * (1 + nu))
        lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
        
        epsilon = lambda u: ufl.sym(ufl.grad(u))
        sigma = lambda u: lambda_ * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)
        
        return ufl.inner(sigma(u), epsilon(v)) * ufl.dx

    
    stiffness_form = truss_stiffness(u, v, E_func, area_func)

    mass_matrix = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(mass_form))
    mass_matrix.assemble()
    
    stiffness_matrix = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(stiffness_form))
    stiffness_matrix.assemble()
    
    return mass_matrix, stiffness_matrix