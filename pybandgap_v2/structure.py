from pybandgap.utility import geometry_utility
from scipy.sparse import csr_matrix
from dataclasses import dataclass
from typing import List, Union
from petsc4py import PETSc
from pybandgap.fem import Fem
import pyvista as pv
import numpy as np
from dolfinx.plot import vtk_mesh

@dataclass
class SetStructure:
    fems: Union[List, Fem]
    geometry: str
    symmetries: Union[list, float]

    def __post_init__(self):
        self.M = None
        self.K = None        

        self.fems = [self.fems] if not isinstance(self.fems, list) else self.fems
        
        self._find_limits()
        self._find_IBZ()
        self._map_structure()
        self._get_boundary_data()
        self._get_Symmetry_Maps()
        self._set_x_structure()
 
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
        
        if self.geometry in ('quadrilateral', 'square'):
            self.geometry_points = [
                [self.x_min, self.y_min, 0],
                [self.x_max, self.y_min, 0],
                [self.x_max, self.y_max, 0],
                [self.x_min, self.y_max, 0],
            ]
            self.Lx = self.x_max - self.x_min
            self.Ly = self.y_max - self.y_min
    
    def _find_IBZ(self):
        angles = self.symmetries['angles']
        directions = self.symmetries['directions']
        
        x0, y0 = self.mid_point[0], self.mid_point[1]
        points = self.geometry_points
        for angle, direction in zip(angles, directions):
            points = geometry_utility.find_intersections_with_line(points, self.mid_point, angle)
            
            mask = np.ones(len(points), dtype=bool)
            x = points[:, 0]
            y = points[:, 1]

            if np.isclose(angle, np.pi/2):
                condition = x >= x0 if direction == "below" else x <= x0
                condition = condition | np.isclose(x - x0, 0)
            elif np.isclose(angle, 3*np.pi/2):
                condition = x <= x0 if direction == "below" else x >= x0
                condition = condition | np.isclose(x - x0, 0)
            else:
                m = np.tan(angle)
                condition = y >= m * (x - x0) + y0 if direction == "above" else y <= m * (x - x0) + y0
                condition = condition | np.isclose(y - m * (x - x0) + y0,0)

            mask = np.logical_and(mask, condition)
            points = points[mask]
        
        self.IBZ_points = np.vstack((points, points[0]))
        
        for fem in self.fems:
            fem.get_elements_in_perimeter(self.IBZ_points)
            
        self._get_elements_IBZ()
            
    def _get_elements_IBZ(self):
        IBZ_elements = {}
        offset = 0
        for i, fem in enumerate(self.fems):
            mesh = fem.mesh
            IBZ_elements[i] = fem.IBZ_elements + offset
            N_elements = mesh.topology.index_map(mesh.topology.dim).size_local
            offset += N_elements
        self.IBZ_elements = IBZ_elements

    def _map_structure(self):
        self._map_nodes_indexes()
        self._map_element_indexes()
        self._get_global_coordinates()

    def _map_nodes_indexes(self):
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
        
    def _map_element_indexes(self):
        global_elements = {}
        self.total_elements = 0
        for i, fem in enumerate(self.fems):
            mesh = fem.mesh
            indexes = np.array(range(mesh.topology.index_map(mesh.topology.dim).size_local)) + self.total_elements
            global_elements[i] = indexes
            self.total_elements += len(indexes)
  
        self.global_elements = global_elements

    def _get_global_coordinates(self):
        """Returns a dictionary mapping global indices to their coordinates."""
        self.global_index_coords = {idx: np.array(list(coord)) for coord, idx in self.global_nodes_coords.items()}

    ### set x props of the structure and
    def _set_x_structure(self):
        self.x = np.array([])
        self.x_map = {}
        offset = 0
        for i, fem in enumerate(self.fems):
            self.x_map[i] = {}
            parameters = fem.parameters
            n_parameters = fem.IBZ_len
            for parameter in parameters:
                self.x = np.hstack((self.x, np.random.random(n_parameters)))
                self.x_map[i][parameter] = np.array(range(n_parameters)) + offset
                offset += n_parameters
        
        self.d_x = np.zeros(len(self.x))
        
    def _get_Symmetry_Maps(self):
        for fem in self.fems:
            fem.get_Symmetry_Map(self.symmetries, self.mid_point)
    
    def apply_x(self):
        for idx, fem in enumerate(self.fems):
            S = fem.S
            for param, index in self.x_map[idx].items():
                data_IBZ = self.x[index]
                fem.parameters[param].x.array[:] = S @ data_IBZ

    ### Boundary conditions and T matrix
    def _get_boundary_data(self):
        if self.geometry in ('quadrilateral', 'square'):
            geometry_utility.quadrilateral_perimeter(self)
        else:
            raise ValueError(f"Geometry {self.geometry} not supported")
        coords_array = np.array(list(self.global_nodes_coords.keys()))
        indices_array = np.array(list(self.global_nodes_coords.values()))
        
        boundary_nodes = {}
        
        for boundary, condition in self.boundary_conditions.items():
            indices = np.where(condition(coords_array))[0]
            if len(indices) > 0:
                boundary_nodes[boundary] = indices_array[indices]
        
        self.boundary_nodes = boundary_nodes

    def T_matrix(self, k_x, k_y):
        num_nodes_x = self.total_nodes
        T_matrix = np.eye(num_nodes_x*2, dtype= np.complex128)
        
        remove_nodes = []
    
        def get_mean_nodes(indexes):
            return np.mean([self.global_index_coords[idx] for idx in indexes], axis=0)
        
        for reference_boundary, boundaries in self.reference_nodes.items():
            reference_nodes = self.boundary_nodes[reference_boundary]
            reference_point = get_mean_nodes(reference_nodes)
            for boundary in boundaries:
                nodes = self.boundary_nodes[boundary]
                remove_nodes.append(nodes)
                boundary_point = get_mean_nodes(nodes)
                L = np.abs(boundary_point - reference_point)
                T_matrix[2*nodes, :] = 0
                T_matrix[2*nodes+1, :] = 0
                T_matrix[2*nodes, 2*reference_nodes] = np.exp(1j *(L[0]*k_x + L[1]*k_y))
                T_matrix[2*nodes+1, 2*reference_nodes+1] = np.exp(1j *(L[0]*k_x + L[1]*k_y))
        
        remove_nodes = np.hstack(remove_nodes)
        remove_columns = np.hstack((2*remove_nodes, 2*remove_nodes+1))
        T_matrix = np.delete(T_matrix, remove_columns, axis=1)
        
        T_csr = csr_matrix(T_matrix)
        
        T = PETSc.Mat().createAIJ(size=(2 * num_nodes_x, T_csr.shape[1]), csr=(T_csr.indptr, T_csr.indices, T_csr.data))
        T.assemble()
        
        self.T = T
    
    ### Assembly of global matrices
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
        def get_values_matrix(matrix):
                vals = matrix.getValues(*matrix.getSize())
                return vals
            
        for local_matrix, node_indices in zip(local_matrices, indices):
            values = get_values_matrix(local_matrix)
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


    def Mass_and_Stiffness_derivate(self, element, variable, fem_idx):
        n_index = self.total_nodes * 2
        dM = self.initialize_global_matrices(n_index)
        dK = self.initialize_global_matrices(n_index)
        
        fem = self.fems[fem_idx]
        
        if variable in fem.parameters:
            dm = fem.get_d_matrix('m', element, variable = variable)
            dk = fem.get_d_matrix('k', element, variable = variable)
        
            indices = fem.global_node_indices
            indices = np.hstack((indices*2, indices*2 + 1))
            dM.setValues(
                    indices, indices, dm[:,:], addv=PETSc.InsertMode.ADD_VALUES
                )

            dK.setValues(
                    indices, indices, dk[:,:], addv=PETSc.InsertMode.ADD_VALUES
                )
            
        dM.assemble()
        dK.assemble()
        
        return dM, dK
    
    def show_structure(self, 
                    colors=None, 
                    background=(31, 31, 31), 
                    highlight_index=None):
        """
        Muestra la estructura con una opción para resaltar un índice específico en verde.

        Args:
            colors (optional): Colores para los elementos.
            background (tuple, optional): Color de fondo de la visualización.
            highlight_index (int, optional): Índice del elemento a resaltar en verde.
        """
        plotter = pv.Plotter()
        plotter.set_background(background)

        for fem in self.fems:
            mesh = fem.mesh
            tdim = mesh.topology.dim
            num_cells_local = mesh.topology.index_map(tdim).size_local
            topology, cell_types, x = vtk_mesh(mesh, tdim, np.arange(num_cells_local, dtype=np.int32))

            grid = pv.UnstructuredGrid(topology, cell_types, x)
            
            if tdim == 1:  # Bar elements
                # Set line width based on `self.parameters['x_A']`
                widths = np.where(fem.parameters['x_A'].x.array == 1, 8, 2)
                grid["line_width"] = widths
                
                # Set colors based on `self.parameters['x_m']`
                colors = np.where(fem.parameters['x_m'].x.array == 1, "red", "blue")
                
                # Override color for the highlighted index
                if highlight_index is not None and 0 <= highlight_index < num_cells_local:
                    colors[highlight_index] = 'w'
                
                grid["color"] = colors
                
                for cell_id in range(num_cells_local):
                    plotter.add_mesh(
                        grid.extract_cells(cell_id),
                        color=grid["color"][cell_id],
                        line_width=grid["line_width"][cell_id],
                        show_edges=True,
                    )
            else:
                # Default visualization for higher-dimensional elements
                plotter.add_mesh(grid, show_edges=True)
            
            plotter.view_xy()

        plotter.show()