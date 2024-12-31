from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Union, Tuple
from pybandgap_opt.utility import MeshUtils, GeometryUtils
from typing import Optional, Tuple

@dataclass
class Material:
    material: str
    young_modulus: float
    poisson_ratio: float
    density: float
    _counter: int = 0
    
    def __post_init__(self):
        self.creation_number = Material._counter
        Material._counter += 1
        
    def __repr__(self):
        return f'{self.material} (Creation #{self.creation_number})'
    
    @classmethod
    def get_counter(cls) -> int:
        return cls._counter

@dataclass
class Props:
    mesh: int
    
    def add_prop(self, name: str, prop: list) -> None:
        setattr(self, name, prop)

@dataclass
class SetStructure:
    meshes: Union[List, Tuple]
    
    def __post_init__(self):
        self.meshes = [self.meshes] if not isinstance(self.meshes, (list, tuple)) else self.meshes
        self.props = [Props(mesh=i) for i in range(len(self.meshes))]
        self._initialize_structure()
    
    def _initialize_structure(self):
        self._find_limits()
        self._map_nodes()
        self._find_boundary_nodes()
    
    def _find_limits(self):
        limits = {axis: {'min': float('inf'), 'max': float('-inf')} 
                 for axis in ['x', 'y', 'z']}
        
        for mesh in self.meshes:
            coords = mesh.geometry.x
            for i, axis in enumerate(['x', 'y', 'z']):
                limits[axis]['min'] = min(limits[axis]['min'], np.min(coords[:, i]))
                limits[axis]['max'] = max(limits[axis]['max'], np.max(coords[:, i]))
        
        for axis, values in limits.items():
            setattr(self, f"{axis}_min", values['min'])
            setattr(self, f"{axis}_max", values['max'])
        
        self.mid_point = [(values['max'] + values['min'])/2 
                         for values in limits.values()]
    
    def _map_nodes(self):
        self.global_indices = []
        current_global_index = 0
        
        first_mesh_size = self.meshes[0].geometry.x.shape[0]
        first_mesh_global = np.arange(first_mesh_size, dtype=np.int32)
        self.global_indices.append(first_mesh_global)
        
        self.node_mapping = {i: i for i in range(first_mesh_size)}
        current_global_index = first_mesh_size
        
        if len(self.meshes) > 1:
            common_indices = MeshUtils.find_common_indices(
                self.meshes[0].geometry.x,
                self.meshes[1].geometry.x
            )
            
            second_mesh_size = self.meshes[1].geometry.x.shape[0]
            second_mesh_global = np.full(second_mesh_size, -1, dtype=np.int32)
            
            for idx1, idx2 in zip(common_indices[0], common_indices[1]):
                second_mesh_global[idx2] = first_mesh_global[idx1]
                self.node_mapping[idx2 + first_mesh_size] = self.node_mapping[idx1]
            
            for i in range(second_mesh_size):
                if second_mesh_global[i] == -1:
                    second_mesh_global[i] = current_global_index
                    self.node_mapping[i + first_mesh_size] = current_global_index
                    current_global_index += 1
            
            self.global_indices.append(second_mesh_global)
            self.total_nodes = current_global_index
        else:
            self.total_nodes = first_mesh_size
    
    def _get_node_indices(self, condition):
        indices = set()
        offset = 0
        
        for mesh_idx, mesh in enumerate(self.meshes):
            mesh_indices = np.where(np.apply_along_axis(condition, 1, mesh.geometry.x))[0]
            
            if mesh_idx == 0:
                indices.update(mesh_indices)
            else:
                indices.update(self.node_mapping[idx + offset] for idx in mesh_indices)
            
            offset += mesh.geometry.x.shape[0]
        
        return np.array(sorted(list(indices)))
    
    def _find_boundary_nodes(self):
        corners = {
            'down_left':  [self.x_min, self.y_min, 0],
            'down_right': [self.x_max, self.y_min, 0],
            'up_left':    [self.x_min, self.y_max, 0],
            'up_right':   [self.x_max, self.y_max, 0]
        }
        
        for name, coords in corners.items():
            setattr(self, f"node_{name}", 
                   self._get_node_indices(lambda x: np.allclose(x, coords)))
        
        boundaries = {
            'bottom': lambda x: np.isclose(x[1], self.y_min) and self.x_min < x[0] < self.x_max,
            'top':    lambda x: np.isclose(x[1], self.y_max) and self.x_min < x[0] < self.x_max,
            'left':   lambda x: np.isclose(x[0], self.x_min) and self.y_min < x[1] < self.y_max,
            'right':  lambda x: np.isclose(x[0], self.x_max) and self.y_min < x[1] < self.y_max
        }
        
        for name, condition in boundaries.items():
            setattr(self, f"node_{name}", self._get_node_indices(condition))
        
        excluded_nodes = sum(len(getattr(self, f"node_{n}")) 
                           for n in ['top', 'right', 'down_right', 'up_right', 'up_left'])
        self.reduced_nodes = self.total_nodes - excluded_nodes
    
    def find_irreducible_brillouin_zone(self, mesh = 0) -> Tuple[np.ndarray, np.ndarray]:
        midpoints = MeshUtils.get_midpoints(self.meshes[mesh])
        
        conditions = np.vstack([
            midpoints[:, 0] >= self.mid_point[0],
            midpoints[:, 1] >= self.mid_point[1],
            (midpoints[:, 1] < midpoints[:, 0]) | 
            np.isclose(midpoints[:, 1], midpoints[:, 0])
        ])
        
        return np.where(np.all(conditions, axis=0))[0], conditions

    def find_symmetry_elements(self, mesh, index: int) -> np.ndarray:
        midpoints = MeshUtils.get_midpoints(self.meshes[mesh])
        point = midpoints[index]
        
        def get_symmetric_index(idx: int, point: np.ndarray, axis: int, 
                              direction: Optional[str] = None) -> np.ndarray:
            indices = np.where(np.all(np.isclose(point, midpoints, atol=1e-10), axis=1))[0]
            
            if len(indices) > 1:
                nodes = MeshUtils.get_nodes(self.meshes[mesh], idx)
                projection = self._apply_symmetry(nodes, axis, direction)
                target_nodes = MeshUtils.get_nodes(self.meshes[mesh], indices[0])
                return indices[0] if np.all(self._rows_match(projection, target_nodes)) else indices[1]
            return indices[0]
        
        result = [index]
        
        if not GeometryUtils.is_in_diagonal(point, self.mid_point):
            diagonal_point, axis = GeometryUtils.symmetric_projection_diagonal(
                self.mid_point, point)
            result.append(get_symmetric_index(index, diagonal_point, axis, 'd'))
        
        for axis in [0, 1]:
            new_indices = []
            for idx in result:
                sym_point = GeometryUtils.symmetric_projection(
                    self.mid_point, midpoints[idx], axis)
                new_indices.append(get_symmetric_index(idx, sym_point, axis))
            result.extend(new_indices)
        
        return np.array(result, dtype=int)
    
    def _apply_symmetry(self, points: np.ndarray, axis: int, 
                       direction: Optional[str] = None) -> np.ndarray:
        if direction == 'd':
            return np.array([GeometryUtils.symmetric_projection_diagonal(
                self.mid_point, p, axis)[0] for p in points])
        return np.array([GeometryUtils.symmetric_projection(
            self.mid_point, p, axis) for p in points])
    
    @staticmethod
    def _rows_match(arr1: np.ndarray, arr2: np.ndarray, tol: float = 1e-8) -> np.ndarray:
        return np.array([
            np.any(np.all(np.isclose(row, arr2, atol=tol, rtol=0), axis=1))
            for row in arr1
        ])
    
    def set_prop(self, name: str, prop: List[Material], mesh: int = 0) -> None:
        self.props[mesh].add_prop(name, self.map_prop(mesh, prop))
    
    def map_prop(self, mesh: int, prop: List[Material]) -> Dict[int, Material]:
        ibz_elements, _ = self.find_irreducible_brillouin_zone(mesh)
        prop_map = dict(zip(ibz_elements, prop))
        
        for idx in ibz_elements:
            symmetric_elements = self.find_symmetry_elements(mesh, idx)
            prop_map.update(dict.fromkeys(symmetric_elements, prop_map[idx]))
        
        return list(dict(sorted(prop_map.items())).values())
    
    def show_structure(self):
        import pyvista
        from dolfinx.plot import vtk_mesh

        colors = [
            [67, 75, 217],
            [217, 67, 70],
            [97, 213, 157],
            [213, 213, 97]
        ]

        p = pyvista.Plotter()
        p.set_background('#1f1f1f')
        p.add_axes(color='w')

        for mesh_i, props_i in zip(self.meshes, self.props):
            tdim = mesh_i.topology.dim
            num_cells_local = mesh_i.topology.index_map(tdim).size_local
            topology, cell_types, x = vtk_mesh(mesh_i, tdim, np.arange(num_cells_local, dtype=np.int32))
            grid = pyvista.UnstructuredGrid(topology, cell_types, x).extract_surface(nonlinear_subdivision=0)

            has_diameters = hasattr(props_i, 'diameters')
            has_materials = hasattr(props_i, 'materials')

            if has_diameters:
                line_widths = np.array([diameter for diameter in props_i.diameters.values()])        
                line_widths = np.exp(line_widths * 250)

            if has_materials:
                marker = np.array([np.array(colors[material.creation_number])/255 
                                   for material in props_i.materials.values()])
                grid.cell_data["colors"] = marker
            
                for cell_id in range(num_cells_local):
                    if has_diameters:
                        p.add_mesh(grid.extract_cells(cell_id), 
                                color=marker[cell_id], 
                                line_width=line_widths[cell_id])
                    else:
                        p.add_mesh(grid.extract_cells(cell_id), 
                                color=marker[cell_id])
            else:
                 p.add_mesh(grid)
        
        p.show_axes()
        p.view_xy()
        p.show()