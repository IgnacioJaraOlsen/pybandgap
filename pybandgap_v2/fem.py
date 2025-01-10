from dataclasses import dataclass
from typing import Callable, Union, Optional, Dict
from dolfinx.fem import Function, functionspace, form
from dolfinx.fem.petsc import  assemble_matrix
from dolfinx.mesh import compute_midpoints
from shapely import Polygon, Point
from pybandgap.utility import geometry_utility
import ufl
import inspect
import numpy as np

@dataclass
class Fem:
    mesh: ufl.Mesh
    props: Dict[str, Optional[Union[float, Callable]]]
    
    def __post_init__(self):
        self.V = functionspace(self.mesh, ("CG", 1, (2,)))
        self.V0 = functionspace(self.mesh, ("DG", 0))
        self.u, self.v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        
        self.N_elements = self.mesh.topology.index_map(self.mesh.topology.dim).size_local
        self._get_parameters()
    
    def _get_parameters(self):
        """
        Extracts parameters and creates a dictionary of Functions.
        - parameters: dictionary mapping parameter names to Function objects
        - props_parameters: dictionary mapping each prop to its parameters
        """
        self.parameters = {}
        self.props_parameters = {}
        
        # Collect unique parameters
        param_set = set()
        for prop_name, prop_value in self.props.items():
            if callable(prop_value) and not isinstance(prop_value, type):
                params = inspect.signature(prop_value).parameters
                param_names = list(params.keys())
                self.props_parameters[prop_name] = param_names
                param_set.update(param_names)
        
        # Create Functions for each parameter
        for param in param_set:
            self.parameters[param] = Function(self.V0)
            
        self.parameters = dict(sorted(self.parameters.items()))
    
    def eval(self, prop_name: str) -> Union[float, Function]:
        """
        Evaluates a property using current parameters.
        
        Args:
            prop_name: Name of the property to evaluate
            
        Returns:
            Result of evaluating the property (float or Function)
            
        Raises:
            ValueError: If the property name is not found
        """
        if prop_name not in self.props:
            raise ValueError(f"Property '{prop_name}' not found")
            
        prop_value = self.props[prop_name]
        
        # If it's a constant value, return it directly
        if not callable(prop_value) or isinstance(prop_value, type):
            return prop_value
            
        # If it's a function, evaluate it with corresponding parameters
        param_names = self.props_parameters[prop_name]
        param_values = [self.parameters[param] for param in param_names]
        
        return prop_value(*param_values)

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

    def get_d_matrix(self, name, element = 0, variable = 'x'):
        elements = Function(self.V0)
        elements.x.array[element] = 1

        if name in ('m','mass'):
            a = self.mass_ufl()
        elif name in ('k', 'stiffness'):
            a = self.stiffness_ufl()

        da = ufl.diff(elements * a * ufl.dx, self.parameters[variable])
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
        index = np.array([i for i, point in enumerate(centers) if polygon.intersects(Point(point))])
        self.IBZ_len = len(index)
        self.IBZ_elements = index

    def get_Symmetry_Map(self, symmetries, center):
        
        indexes = self.IBZ_elements
        
        angles = symmetries['angles'][::-1]
        elements = np.array(range(self.N_elements))
        
        def f_id_element(idx):
            nodes = np.sort(self.get_nodes_element(idx), axis=0).ravel()
            id = nodes
            # Add center point
            centers = self.get_midpoints()[idx]
            id = np.hstack((id, centers))

            if self.mesh.topology.dim == 1:
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
            if self.mesh.topology.dim == 1:
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
        
        self.S = S