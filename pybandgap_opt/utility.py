import numpy as np
from dolfinx.mesh import compute_midpoints
from typing import Optional, Tuple

class MeshUtils:
    @staticmethod
    def get_midpoints(mesh) -> np.ndarray:
        tdim = mesh.topology.dim
        mesh_entities = mesh.topology.index_map(tdim).size_local
        return compute_midpoints(mesh, tdim, np.arange(mesh_entities))
    
    @staticmethod
    def get_nodes(mesh, index: int) -> np.ndarray:
        return mesh.geometry.x[mesh.topology.connectivity(mesh.topology.dim, 0).links(index)]
    
    @staticmethod
    def find_common_indices(array1: np.ndarray, array2: np.ndarray, tol: float = 1e-10) -> np.ndarray:
        decimals = int(-np.log10(tol))
        array1_dict = {tuple(np.round(row, decimals)): i for i, row in enumerate(array1)}
        
        indices = [], []
        for idx2, row in enumerate(array2):
            row_tuple = tuple(np.round(row, decimals))
            if row_tuple in array1_dict:
                idx1 = array1_dict[row_tuple]
                if np.allclose(array1[idx1], row, atol=tol):
                    indices[0].append(idx1)
                    indices[1].append(idx2)
        
        return np.vstack([np.array(idx) for idx in indices])

class GeometryUtils:
    @staticmethod
    def symmetric_projection(center: np.ndarray, point: np.ndarray, axis: int = 0) -> np.ndarray:
        result = point.copy()
        axis = 1 - axis
        result[axis] += 2 * (center[axis] - point[axis])
        return result
    
    @staticmethod
    def symmetric_projection_diagonal(p1: np.ndarray, p2: np.ndarray, 
                                   axis: Optional[int] = None) -> Tuple[np.ndarray, int]:
        relative = p2 - p1
        x, y = relative[0], relative[1]
        
        if axis is None:
            axis = 1 if abs(y - x) <= abs(y + x) else -1
            
        symmetric = np.array([y, x, relative[2]]) if axis == 1 else \
                   np.array([-y, -x, relative[2]]) if axis == -1 else \
                   None
                   
        if symmetric is None:
            raise ValueError("axis must be None, 1 or -1")
            
        return symmetric + p1, axis
    
    @staticmethod
    def is_in_diagonal(point: np.ndarray, midpoint: np.ndarray, tol: float = 1e-10) -> bool:
        relative = point - midpoint
        return np.isclose(relative[0], relative[1], atol=tol) or \
               np.isclose(-relative[0], relative[1], atol=tol)