from dataclasses import dataclass
from typing import Callable, Union, Optional, Dict
from dolfinx.fem import Function, functionspace, form
from dolfinx.fem.petsc import  assemble_matrix
from dolfinx.mesh import compute_midpoints
from shapely import Polygon, Point
from utility import geometry_utility
import ufl
import inspect
import numpy as np

#np.set_printoptions(precision=3, suppress=False, formatter={'float': '{:.3f}'.format})

@dataclass
class Fem:
    mesh: ufl.Mesh
    parameters: Dict[str, Optional[Union[float, Callable]]] = None

    def __post_init__(self):
        self.V = functionspace(self.mesh, ("CG", 1, (2,)))
        self.V0 = functionspace(self.mesh, ("DG", 0))
        self.u, self.v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        self.props = {}
    
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
                self.props[name] = param_dict
    
    def eval(self, name):
        if name not in self.parameters.keys():
            return 1
        value = self.parameters.get(name)
        if not callable(value):
            return value
        param_dict = self.props[name]
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
        param_func = self.props[param][variable]

        if name in ('m','mass'):
            a = self.mass_ufl()
        elif name in ('k', 'stiffness'):
            a = self.stiffness_ufl()

        da = ufl.diff(elements * a * ufl.dx, param_func)
        d_matrix = assemble_matrix(form(da))
        d_matrix.assemble()
        return d_matrix
    
    def get_nodes_element(self, index):
        mesh = self.mesh
        return mesh.geometry.x[mesh.topology.connectivity(mesh.topology.dim, 0).links(index)]

    def get_midpoints(self):
        mesh = self.mesh
        tdim = mesh.topology.dim
        mesh_entities = mesh.topology.index_map(tdim).size_local
        return compute_midpoints(mesh, tdim, np.arange(mesh_entities))

    def get_elements_in_perimeter(self, perimeter):
        polygon = Polygon(perimeter)
        centers = self.get_midpoints()
        index = [i for i, point in enumerate(centers) if polygon.intersects(Point(point))]
        self.IBZ_elements = index
        return index

    def get_Symmetry_Map(self, symmetries, center):
        
        indexes = self.IBZ_elements
        
        angles = symmetries['angles'][::-1]
        elements = np.array(range(self.mesh.topology.index_map(self.mesh.topology.dim).size_local))
        
        def f_id_element(idx):
            nodes = np.sort(self.get_nodes_element(idx), axis=0).ravel()
            id = nodes
            
            # Add center point
            centers = self.get_midpoints()[idx]
            id = np.hstack((id, centers))
                
            # Add slope
            slopes = geometry_utility.calculate_slope(self.get_nodes_element(idx))
            id = np.hstack((id, slopes))
            
            return id
        
        def calculate_symmetric_id(coords_element, angle):
            # Project nodes
            f = lambda x: geometry_utility.calculate_symmetric_point(x, center, angle)
            nodes = np.sort(np.array(list(map(f, coords_element))), axis=0).ravel()
            id = nodes
            
            # Project and add center point
            center_point = np.mean(coords_element, axis=0)
            symmetric_center = f(center_point)
            id = np.hstack((id, symmetric_center))
            
            # Calculate and add slope of projected element
            symmetric_coords = np.array(list(map(f, coords_element)))
            symmetric_slope = geometry_utility.calculate_slope(symmetric_coords)
            id = np.hstack((id, symmetric_slope))
            
            return id
        
        id_elements = np.array(list(map(f_id_element, elements)))
        
        def arrays_match(arr1, arr2, rtol=1e-10, atol=1e-10):
            if arr1.shape != arr2.shape:
                return False
            return np.all(np.isclose(arr1, arr2, rtol=rtol, atol=atol))
        
        S = []
        
        for idx in indexes:
            elements = [idx]
            for angle in angles:
                new_elements = []
                for element in elements:
                    coords_element = self.get_nodes_element(element)
                    id_element = calculate_symmetric_id(coords_element, angle)
                    new_idx = [i for i, row in enumerate(id_elements) if arrays_match(row, id_element)]
                    new_elements.extend(new_idx)
                elements.extend(new_elements)
                new_elements = []
            
            column = np.zeros(len(id_elements), dtype=int)
            column[np.array(elements)] = 1
            
            S.append(column.reshape(-1, 1))
        
        S = np.hstack(S)
        
        return S