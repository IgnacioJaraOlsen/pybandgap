from dataclasses import dataclass
from dolfinx.mesh import compute_midpoints
import numpy as np
from typing import List, Optional, Dict, Union, Tuple

@dataclass
class Material:
    material: str
    young_modulus: float
    poisson_ratio: float
    density: float
    _counter: int = -1
    
    def __post_init__(self):
        Material._counter += 1
        self.creation_number = Material._counter
        
    def __repr__(self):
        return f'{self.material} (Creation Number: {self.creation_number})'
    
    @classmethod
    def get_counter(cls) -> int:
        return cls._counter

@dataclass
class Props:
    mesh: int
    
    def add_prop(self, name:str, prop: list):
        setattr(self, name, prop)
    
@dataclass
class SetStructure:
    meshes: Union[List, Tuple]
    
    def __post_init__(self):
        self.meshes = [self.meshes] if not isinstance(self.meshes, (list, tuple)) else self.meshes
        self.props = [Props(mesh = i) for i in range(len(self.meshes))]
        
        self._find_limits()
        self._map_nodes()
        self._find_boundary_nodes()

    def _find_limits(self):
        """Calculate global coordinate limits across all meshes"""
        limits = {
            'x': {'min': float('inf'), 'max': float('-inf')},
            'y': {'min': float('inf'), 'max': float('-inf')},
            'z': {'min': float('inf'), 'max': float('-inf')}
        }
        
        for mesh in self.meshes:
            coords = mesh.geometry.x
            for i, axis in enumerate(['x', 'y', 'z']):
                limits[axis]['min'] = min(limits[axis]['min'], np.min(coords[:, i]))
                limits[axis]['max'] = max(limits[axis]['max'], np.max(coords[:, i]))
        
        for axis, values in limits.items():
            setattr(self, f"{axis}_min", values['min'])
            setattr(self, f"{axis}_max", values['max'])
        
        self.mid_point = [
            (limits['x']['max'] + limits['x']['min'])/2,
            (limits['y']['max'] + limits['y']['min'])/2,
            (limits['z']['max'] + limits['z']['min'])/2
        ]
    
    def _map_nodes(self):
        """Map common node indices across meshes"""
        node_mapping = {i: i for i in range(self.meshes[0].shape[0])}
        current_index = self.meshes[0].shape[0]
        
        if len(self.meshes) > 1:
            common_indices = self._find_common_indices(self.meshes[0], self.meshes[1])
            offset = self.meshes[0].shape[0]
            
            for i in range(self.meshes[1].shape[0]):
                if i in common_indices[1]:
                    idx = np.where(common_indices[1] == i)[0][0]
                    node_mapping[i + offset] = node_mapping[common_indices[0][idx]]
                else:
                    node_mapping[i + offset] = current_index
                    current_index += 1
                    
            self.total_nodes = sum(m.shape[0] for m in self.meshes) - len(common_indices[0])
        else:
            self.total_nodes = self.meshes[0].shape[0]
            
        self.node_mapping = node_mapping
    
    def _get_node_indices(self, condition):
        """Get node indices that satisfy a given condition across all meshes"""
        indices = set()
        offset = 0
        
        for mesh_idx, mesh in enumerate(self.meshes):
            mesh_coords = mesh.geometry.x
            mesh_indices = np.where(np.apply_along_axis(condition, 1, mesh_coords))[0]
            
            if mesh_idx == 0:
                indices.update(mesh_indices)
            else:
                mapped_indices = [self.node_mapping[idx + offset] for idx in mesh_indices]
                indices.update(mapped_indices)
            
            offset += mesh.shape[0]
        
        return np.array(sorted(list(indices)))
    
    def _find_boundary_nodes(self):
        """Identify boundary nodes: corners and perimeter"""
        corners = {
            'down_left':  [self.x_min, self.y_min, 0],
            'down_right': [self.x_max, self.y_min, 0],
            'up_left':    [self.x_min, self.y_max, 0],
            'up_right':   [self.x_max, self.y_max, 0]
        }
        
        # Find corner nodes
        for name, coords in corners.items():
            setattr(self, f"node_{name}", 
                   self._get_node_indices(lambda x: np.allclose(x, coords)))
        
        # Find perimeter nodes
        boundaries = {
            'bottom': lambda x: np.isclose(x[1], self.y_min) and self.x_min < x[0] < self.x_max,
            'top':    lambda x: np.isclose(x[1], self.y_max) and self.x_min < x[0] < self.x_max,
            'left':   lambda x: np.isclose(x[0], self.x_min) and self.y_min < x[1] < self.y_max,
            'right':  lambda x: np.isclose(x[0], self.x_max) and self.y_min < x[1] < self.y_max
        }
        
        for name, condition in boundaries.items():
            setattr(self, f"node_{name}", self._get_node_indices(condition))
        
        excluded_nodes = sum(len(getattr(self, f"node_{n}")) for n in ['top', 'right', 
                           'down_right', 'up_right', 'up_left'])
        self.reduced_nodes = self.total_nodes - excluded_nodes
    
    @staticmethod
    def _find_common_indices(array1, array2, tol=1e-10):
        """Find common indices between two arrays within tolerance"""
        decimals = int(-np.log10(tol))
        array1_dict = {tuple(np.round(row, decimals=decimals)): i 
                      for i, row in enumerate(array1)}
        
        common_indices = [], []
        for idx2, row in enumerate(array2):
            row_tuple = tuple(np.round(row, decimals=decimals))
            if row_tuple in array1_dict:
                idx1 = array1_dict[row_tuple]
                if np.allclose(array1[idx1], row, atol=tol):
                    common_indices[0].append(idx1)
                    common_indices[1].append(idx2)
        
        return np.vstack([np.array(indices) for indices in common_indices])

    def find_irreducible_brillouin_zone(self, mesh) -> Tuple[np.ndarray, np.ndarray]:
        """Find irreducible Brillouin zone elements"""
        midpoint = self.mid_point
        midpoints = get_midpoint_elements(self.meshes[mesh])
        
        conditions = np.vstack([
            midpoints[:, 0] >= midpoint[0],
            midpoints[:, 1] >= midpoint[1],
            (midpoints[:, 1] < midpoints[:, 0]) | np.isclose(midpoints[:, 1], midpoints[:, 0])
        ])
        
        return np.where(np.all(conditions, axis=0))[0], conditions

    def find_symmetry_elements(self, mesh, index: int) -> np.ndarray:
        """Find symmetric elements for given mesh element"""
        midpoint = self.mid_points
        midpoints = get_midpoint_elements(self.meshes[mesh])
        point = midpoints[index]
        
        def get_symmetric_index(idx: int, point: np.ndarray, axis: int, 
                            direction: Optional[str] = None) -> np.ndarray:
            indices = np.where(np.all(np.isclose(point, midpoints, atol=1e-10), axis=1))[0]
            
            if len(indices) > 1:
                nodes = get_nodes_element(mesh, idx)
                projection = apply_symmetric_projection(midpoint, nodes, axis, direction)
                return indices[0] if rows_in_array(projection, 
                    get_nodes_element(mesh, indices[0])).all() else indices[1]
            return indices[0]
        
        result = [index]
        
        if not is_in_diagonal(point, midpoint):
            diagonal_point, axis = symmetric_projection_diagonal(midpoint, point)
            result.append(get_symmetric_index(index, diagonal_point, axis, 'd'))
        
        # X and Y symmetries
        for axis in [0, 1]:
            new_indices = []
            for idx in result:
                sym_point = symmetric_projection(midpoint, midpoints[idx], axis)
                new_indices.append(get_symmetric_index(idx, sym_point, axis))
            result.extend(new_indices)
        
        return np.array(result, dtype=int)

    def set_prop(self, name: str, prop: List[Material], mesh = 0) -> None:
        prop = self.props[mesh]
        prop.add_prop(name, self.map_prop(self.meshes[mesh], prop))

    def map_prop(self, mesh, prop: List[Material]) -> Dict[int, Material]:
        """Map properties to mesh elements considering symmetry"""
        ibz_elements, _ = self.find_irreducible_brillouin_zone(mesh)
        prop_map = dict(zip(ibz_elements, prop))
        
        for idx in ibz_elements:
            symmetric_elements = self.find_symmetry_elements(mesh, idx)
            prop_map.update(dict.fromkeys(symmetric_elements, prop_map[idx]))
        
        return dict(sorted(prop_map.items()))

def get_midpoint_elements(mesh):
    """Calculate midpoints of mesh elements"""
    tdim = mesh.topology.dim
    mesh_entities = mesh.topology.index_map(tdim).size_local
    return compute_midpoints(mesh, tdim, np.arange(mesh_entities))

def symmetric_projection(center: np.ndarray, point: np.ndarray, axis: int = 0) -> np.ndarray:
    """Project point symmetrically across axis"""
    result = point.copy()
    axis = 1 - axis  # Flip axis
    result[axis] += 2 * (center[axis] - point[axis])
    return result

def symmetric_projection_diagonal(p1: np.ndarray, p2: np.ndarray, axis: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Project point symmetrically across diagonal"""
    relative = p2 - p1
    x, y = relative[0], relative[1]
    
    if axis is None:
        axis = 1 if abs(y - x) <= abs(y + x) else -1
    
    if axis == 1:
        symmetric = np.array([y, x, relative[2]])
    elif axis == -1:
        symmetric = np.array([-y, -x, relative[2]])
    else:
        raise ValueError("axis must be None, 1 or -1")
        
    return symmetric + p1, axis

def apply_symmetric_projection(center: np.ndarray, points: np.ndarray, 
                             axis: int, direction: Optional[str] = None) -> np.ndarray:
    """Apply symmetric projection to multiple points"""
    if direction == 'd':
        return np.array([symmetric_projection_diagonal(center, p, axis)[0] for p in points])
    return np.array([symmetric_projection(center, p, axis) for p in points])

def get_nodes_element(mesh, index: int) -> np.ndarray:
    """Get node coordinates for mesh element"""
    return mesh.geometry.x[mesh.topology.connectivity(mesh.topology.dim, 0).links(index)]

def rows_in_array(arr1: np.ndarray, arr2: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    """Check if rows from arr1 exist in arr2"""
    return np.array([
        np.any(np.all(np.isclose(row, arr2, atol=tolerance, rtol=0), axis=1))
        for row in arr1
    ])

def is_in_diagonal(point: np.ndarray, midpoint: np.ndarray, tolerance: float = 1e-10) -> bool:
    """Check if point lies on diagonal"""
    relative = point - midpoint
    return np.isclose(relative[0], relative[1], atol=tolerance) or \
           np.isclose(-relative[0], relative[1], atol=tolerance)
           
