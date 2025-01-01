from dataclasses import dataclass
from typing import List, Dict, Union, Tuple, Optional
from fem import Fem
import numpy as np

@dataclass
class SetStructure:
    fems: Union[List, Fem]
    geometry: str

    def __post_init__(self):
        self.fems = [self.fems] if not isinstance(self.fems, list) else self.fems
        self._find_limits()
        self._assign_global_indices()

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

    def get_global_coordinates(self):
        """Returns a dictionary mapping global indices to their coordinates."""
        return {idx: np.array(coord) for coord, idx in self.global_nodes_coords.items()}

if __name__ == '__main__':
    from meshes.truss_like_mesh import truss_like_cross

    def create_linear_function(a1, a2):
        """Create a linear function between two values."""
        return lambda x: (a2 - a1) * x + a1

    mesh, *_ = truss_like_cross()

    A_funtion =  create_linear_function(np.pi*4e-6,
                                        np.pi*16e-6)

    E_funtion =  create_linear_function(70e9,
                                        411e9)

    rho_funtion =  create_linear_function(2.7e3,
                                        19.3e3)
    
    fem = Fem(mesh, {'A':A_funtion,
                     'E': E_funtion,
                     'rho': rho_funtion})
    
    structure = SetStructure(fem,
                             'square')

    print(f'mid pont: {tuple(float(x) for x in structure.mid_point)}')