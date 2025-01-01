from dataclasses import dataclass
from typing import Callable, Union, Optional, Dict
from dolfinx.fem import Function, functionspace, form
from dolfinx.fem.petsc import  assemble_matrix, assemble_vector
import ufl
import inspect
import numpy as np
from mpi4py import MPI

np.set_printoptions(precision=3, suppress=False, formatter={'float': '{:.2f}'.format})

@dataclass
class Fem:
    mesh: ufl.Mesh
    parameters: Dict[str, Optional[Union[float, Callable]]] = None

    def __post_init__(self):
        self.V = functionspace(self.mesh, ("CG", 1, (2,)))
        self.V0 = functionspace(self.mesh, ("DG", 0))
        self.u, self.v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
    
        parameter_names = self.parameters.keys()

        for name in parameter_names:
            value = self.parameters.get(name)
            if callable(value):
                signature = inspect.signature(value)
                parameters = signature.parameters
                param_dict = {
                    param_name: Function(self.V0)
                    for param_name, param in parameters.items()
                    if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                }
                setattr(self, name, param_dict)
    
    def eval(self, name):
        if name not in self.parameters.keys():
            return 1
        value = self.parameters.get(name)
        if not callable(value):
            return value
        param_dict = getattr(self, name)
        return value(*param_dict.values())

    def mass_ufl(self):
        rho = self.eval('rho')
        if self.mesh.topology.dim == 1:
            A = self.eval('A')
            return A * rho * ufl.inner(self.u, self.v)
        else:
            thickness = self.eval('thickness')
            return thickness * rho * ufl.inner(self.u, self.v)

    
    def stiffness_ufl(self):
        E = self.eval('E')
        epsilon = lambda u: ufl.sym(ufl.grad(u))
        
        if self.mesh.topology.dim == 1:
            A = self.eval('A')
            sigma = lambda u: A * E * ufl.tr(epsilon(u)) * ufl.Identity(len(u))
        else:
            nu = self.eval('nu')
            mu = E / (2 * (1 + nu))
            lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
            sigma = lambda u: lambda_ * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2.0 * mu * epsilon(u)
        
        return ufl.inner(sigma(self.u), epsilon(self.v))

    def get_matrix(self, name):
        name = name.lower()
        if name in ('m','mass'):
            a = self.mass_ufl() * ufl.dx
        elif name in ('k', 'stiffness'):
            a = self.stiffness_ufl() * ufl.dx
        
        matrix = assemble_matrix(form(a))
        matrix.assemble()
        return matrix
    
    def get_d_matrix(self, name, param, element = 0, variable = 'x'):
        elements = Function(self.V0)
        elements.x.array[element] = 1
        param_func = getattr(self, f"{param}", None)[variable]

        if name in ('m','mass'):
            a = self.mass_ufl()
        elif name in ('k', 'stiffness'):
            a = self.stiffness_ufl()

        da = ufl.diff(elements * a * ufl.dx, param_func)
        d_matrix = assemble_matrix(form(da))
        d_matrix.assemble()
        return d_matrix


if __name__ == '__main__':

    def create_linear_function(a1: float, a2: float) -> Callable:
        """Create a linear function between two values."""
        return lambda x: (a2 - a1) * x + a1
    
    import dolfinx
    from mpi4py import MPI
    from dolfinx import io
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0) 

    p1 = gmsh.model.geo.addPoint(0, 0, 0, 1.0) # x, y, z, tamaño característico
    p2 = gmsh.model.geo.addPoint(1, 0, 0, 1.0)
    p3 = gmsh.model.geo.addPoint(2, 0, 0, 1.0)
    linea = gmsh.model.geo.addLine(p1, p2)
    linea2 = gmsh.model.geo.addLine(p2, p3)

    gmsh.model.geo.synchronize()

    gmsh.model.add_physical_group(1, [linea, linea2], 2)

    # Sincronizar para aplicar los cambios
    gmsh.model.geo.synchronize()
    for line in gmsh.model.getEntities(1):
        line_tag = line[1]
        gmsh.model.mesh.set_transfinite_curve(line_tag, 2)

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(dim=1)

    mesh, markers, facets = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)

    gmsh.finalize()

    print(type(mesh))

    E_func = create_linear_function(a1=10, a2=20)
    A_func = create_linear_function(a1=1, a2=5)

    fem = Fem(mesh, parameters = {'E': E_func,
                                  'A': A_func})
    fem.E['x'].x.array[:] = 1
    fem.A['x'].x.array[:] = 0

    # Get the stiffness and mass matrices
    stiffness_matrix = fem.get_matrix('k')
    mass_matrix = fem.get_matrix('m')

    print("Stiffness matrix:")
    print(stiffness_matrix[:,:])

    print("Mass matrix:")
    print(mass_matrix[:,:])

    dmass_matrix = fem.get_d_matrix('k','E', 1)

    print("dMass matrix:")
    print(dmass_matrix[:,:])