from dataclasses import dataclass, field
from pybandgap.multi_mesh import find_common_indices
import numpy as np

@dataclass
class Mesh:
    meshes: list | tuple
    props: list | tuple
    
    def __post_init__(self):
        # Ensure meshes and props are arrays
        if not isinstance(self.meshes, (list, tuple)):
            self.meshes = [self.meshes]
        
        if not isinstance(self.props, (list, tuple)):
            self.props = [self.props]
       
        # find global limits
        self.find_limits()
    
        # map nodes
        self.map_node()
        
        # find boundary nodes
        self.find_boundary_nodes()

    def find_limits(self):
        # find global limits
        self.x_min = float('inf')
        self.x_max = float('-inf')
        self.y_min = float('inf')
        self.y_max = float('-inf')
        
        for mesh in self.meshes:
            x = mesh.geometry.x
            self.x_min = min(self.x_min, np.min(x[:, 0]))
            self.x_max = max(self.x_max, np.max(x[:, 0]))
            self.y_min = min(self.y_min, np.min(x[:, 1]))
            self.y_max = max(self.y_max, np.max(x[:, 1]))        
    
    def map_node(self):
        # map common indices and find total nodes
        node_mapping = {}
        current_index = 0
            
        for i in range(self.meshes[0].shape[0]):
            node_mapping[i] = current_index
            current_index += 1

        if len(self.meshes) > 1:
            index = find_common_indices(self.meshes[0], self.meshes[1])
            self.total_nodes = sum([x.shape[0] for x in self.meshes]) - len(index[0])

            offset = self.meshes[0].shape[0]
            for i in range(self.meshes[1].shape[0]):
                if i in index[1]:
                    idx = np.where(index[1] == i)[0][0]
                    node_mapping[i + offset] = node_mapping[index[0][idx]]
                else:
                    node_mapping[i + offset] = current_index
                    current_index += 1

        else:
            self.total_nodes = self.meshes[0].shape[0]
            
        self.node_mapping = node_mapping        
    
    def get_node_indices(self, condition):
        indices = set()
        for mesh_idx, mesh in enumerate(self.meshes):
            x = mesh.geometry.x
            mesh_indices = np.where(np.apply_along_axis(condition, 1, x))[0]
            if mesh_idx == 0:
                indices.update(mesh_indices)
            else:
                offset = sum([x.shape[0] for x in self.meshes[:mesh_idx]])
                mapped_indices = [self.node_mapping[idx + offset] for idx in mesh_indices]
                indices.update(mapped_indices)
        
        return np.array(sorted(list(indices)))
    
    def find_boundary_nodes(self):    
        # find corner nodes
        down_left   = [self.x_min, self.y_min, 0]
        down_right  = [self.x_max, self.y_min, 0]
        up_left     = [self.x_min, self.y_max, 0]
        up_right    = [self.x_max, self.y_max, 0]
        
        self.node_down_left = self.get_node_indices(lambda coord: np.allclose(coord, down_left))
        self.node_down_right = self.get_node_indices(lambda coord: np.allclose(coord, down_right))
        self.node_up_left = self.get_node_indices(lambda coord: np.allclose(coord, up_left))
        self.node_up_right = self.get_node_indices(lambda coord: np.allclose(coord, up_right))        
        
        # find perimeter nodes
        top = self.y_max
        bottom = self.y_min
        left = self.x_min
        right = self.x_max
        
        self.node_bottom = self.get_node_indices(lambda coord: np.isclose(coord[1], bottom) and coord[0] < right and coord[0] > left)
        self.node_top = self.get_node_indices(lambda coord: np.isclose(coord[1], top) and coord[0] < right and coord[0] > left)
        self.node_left = self.get_node_indices(lambda coord: np.isclose(coord[0], left) and coord[1] < top and coord[1] > bottom)
        self.node_right = self.get_node_indices(lambda coord: np.isclose(coord[0], right) and coord[1] < top and coord[1] > bottom)
        
        self.total_nodes -= len(self.node_top) + len(self.node_right) + len(self.node_down_right) + len(self.node_up_right) + len(self.node_up_left)