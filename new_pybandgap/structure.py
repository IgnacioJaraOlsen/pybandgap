from utility import geometry_utility
from dataclasses import dataclass
from typing import List, Union
from petsc4py import PETSc
from fem import Fem
import numpy as np

class MatExtended(PETSc.Mat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def conjugate_transpose(self):
        mat_transpose = self.transpose()
        mat_conjugate_transpose = mat_transpose.conjugate()
        return mat_conjugate_transpose

@dataclass
class SetStructure:
    fems: Union[List, Fem]
    geometry: str
    projections: Union[list, float]

    def __post_init__(self):
        self.M = None
        self.K = None        

        self.fems = [self.fems] if not isinstance(self.fems, list) else self.fems
        self.projections = [self.projections] if not isinstance(self.projections, list) else self.projections
        
        self._find_limits()
        self._assign_global_indices()

    def _get_global_coordinates(self):
        """Returns a dictionary mapping global indices to their coordinates."""
        return {idx: np.array(coord) for coord, idx in self.global_nodes_coords.items()}

    def _assign_global_indices(self):
        """Assigns global indices to nodes across all meshes."""
        # Dictionary to store unique node coordinates and their global indices
        global_nodes = {}
        current_global_index = 0
        
        # Tolerance for floating-point comparison
        tol = 1e-10
        
        # First pass: identify unique nodes and assign global indices
        for fem in self.fems:
            coords = fem.mesh.geometry.x
            local_to_global = np.zeros(len(coords), dtype=np.int32)
            
            for local_idx, coord in enumerate(coords):
                # Convert coordinate to tuple for dictionary key
                coord_tuple = tuple(coord)
                
                # Check if this node already exists (within tolerance)
                found = False
                for existing_coord, global_idx in global_nodes.items():
                    if np.all(np.abs(np.array(existing_coord) - coord) < tol):
                        local_to_global[local_idx] = global_idx
                        found = True
                        break
                
                if not found:
                    global_nodes[coord_tuple] = current_global_index
                    local_to_global[local_idx] = current_global_index
                    current_global_index += 1
            
            # Store the mapping in the Fem object
            fem.global_node_indices = local_to_global
        
        self.total_nodes = current_global_index
        self.global_nodes_coords = global_nodes

    def _find_limits(self):
        limits = {axis: {'min': float('inf'), 'max': float('-inf')} 
                 for axis in ['x', 'y', 'z']}
        
        for fem in self.fems:
            mesh = fem.mesh
            coords = mesh.geometry.x
            for i, axis in enumerate(['x', 'y', 'z']):
                limits[axis]['min'] = min(limits[axis]['min'], np.min(coords[:, i]))
                limits[axis]['max'] = max(limits[axis]['max'], np.max(coords[:, i]))
        
        for axis, values in limits.items():
            setattr(self, f"{axis}_min", values['min'])
            setattr(self, f"{axis}_max", values['max'])
        
        self.mid_point = [(values['max'] + values['min'])/2
                         for values in limits.values()]

        self.Lx = self.x_max - self.x_min
        self.Ly = self.y_max - self.y_min

    ### Boundary conditions and T matrixs
    def get_mean_nodes(self, indexes):
        return np.mean([self.get_global_coordinates()[idx] for idx in indexes], axis=0)

    def get_boundary_data(self):
        if self.geometry in ('quadrilateral', 'square'):
            boundary_conditions, reference_nodes = geometry_utility.quadrilateral_perimeter(self)
        else:
            raise ValueError(f"Geometry {self.geometry} not supported")
        coords_array = np.array(list(self.global_nodes_coords.keys()))
        indices_array = np.array(list(self.global_nodes_coords.values()))
        
        # Diccionarios para almacenar los resultados
        boundary_nodes = {}
        
        for boundary, condition in boundary_conditions.items():
            indices = np.where(condition(coords_array))[0]
            if len(indices) > 0:
                boundary_nodes[boundary] = indices_array[indices]
        
        return {
            'reference': reference_nodes,
            'boundary': boundary_nodes,
        }          

    def T_matrix(self,  k_x, k_y):
        boundary_data = self.get_boundary_data()
        num_nodes_x = self.total_nodes
        T_matrix = np.eye(num_nodes_x*2)
        
        remove_nodes = []
        for reference_boundary, boundaries in boundary_data['reference'].items():
            reference_nodes = boundary_data['boundary'][reference_boundary]
            reference_point = self.get_mean_nodes(reference_nodes)
            
            for boundary in boundaries:
                nodes = boundary_data['boundary'][boundary]
                remove_nodes.append(nodes)
                boundary_point = self.get_mean_nodes(nodes)
                L = boundary_point - reference_point
                T_matrix[2*nodes, :] = 0
                T_matrix[2*nodes+1, :] = 0
                T_matrix[2*nodes, 2*reference_nodes] = np.exp(1j *(L[0]*k_x + L[1]*k_y))
                T_matrix[2*nodes + 1, 2*reference_nodes + 1] = np.exp(1j *(L[0]*k_x + L[1]*k_y))
        
        remove_nodes = np.hstack(remove_nodes)
        remove_columns = np.hstack((2*remove_nodes, 2*remove_nodes+1))
        T_matrix = np.delete(T_matrix, remove_columns, axis=1)
        
        num_nodes_y = num_nodes_x - len(remove_nodes)
        
        # Create T matrix
        T = MatExtended().create()
        T.setSizes((2 * num_nodes_x, 2 * num_nodes_y))
        T.setType(PETSc.Mat.Type.AIJ)
        T.setUp()
        
        T.setValues(range(2 * num_nodes_x), range(2 * num_nodes_y), T_matrix)

        T.assemblyBegin()
        T.assemblyEnd()
        self.T = T


    ### Assembly of global matrices
    @staticmethod
    def get_values_matrix(matrix):
        vals = matrix.getValues(*matrix.getSize())
        return vals

    def initialize_global_matrices(self, size):
        """
        Create and initialize global PETSc matrices for Mass and Stiffness.
        """
        matrix = PETSc.Mat().create()
        matrix.setSizes((size, size))
        matrix.setType(PETSc.Mat.Type.AIJ)
        matrix.setUp()
        return matrix

    def assemble_global_matrix(self, global_matrix, local_matrices, indices):
        """
        Assemble a global PETSc matrix by adding contributions from local matrices.
        """
        for local_matrix, node_indices in zip(local_matrices, indices):
            values = self.get_values_matrix(local_matrix)
            global_matrix.setValues(
                node_indices, node_indices, values, addv=PETSc.InsertMode.ADD_VALUES
            )
        global_matrix.assemblyBegin()
        global_matrix.assemblyEnd()

    def Mass_and_Stiffness_matrix(self):
        """
        Assemble global Mass (M) and Stiffness (K) matrices.
        """
        local_mass_matrices = [fem.get_matrix('m') for fem in self.fems]
        local_stiffness_matrices = [fem.get_matrix('k') for fem in self.fems]

        if len(self.fems) == 1:
            self.M = local_mass_matrices[0]
            self.K = local_stiffness_matrices[0]
        else:
            n_index = self.total_nodes * 2
            self.M = self.initialize_global_matrices(n_index)
            self.K = self.initialize_global_matrices(n_index)

            global_indices = [
                np.hstack((fem.global_node_indices * 2, fem.global_node_indices * 2 + 1))
                for fem in self.fems
            ]

            self.assemble_global_matrix(self.M, local_mass_matrices, global_indices)
            self.assemble_global_matrix(self.K, local_stiffness_matrices, global_indices)