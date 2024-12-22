import numpy as np
from petsc4py import PETSc
from pybandgap.multi_mesh import find_common_indices

class MatExtended(PETSc.Mat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def conjugate_transpose(self):
        # Get the transpose of the matrix
        mat_transpose = self.transpose()
        
        # Apply conjugation to the transpose
        mat_conjugate_transpose = mat_transpose.conjugate()
        
        return mat_conjugate_transpose

def T_matrix(meshes):
    if not isinstance(meshes, (list, tuple)):
        meshes = [meshes]
        
    # Encontrar límites globales y nodos únicos
    x_min = float('inf')
    x_max = float('-inf')
    y_min = float('inf')
    y_max = float('-inf')
    
    # Recolectar todas las coordenadas
    all_coords = []
    sizes = []
    for mesh in meshes:
        x = mesh.geometry.x
        all_coords.append(x)
        sizes.append(x.shape[0])
        x_min = min(x_min, np.min(x[:, 0]))
        x_max = max(x_max, np.max(x[:, 0]))
        y_min = min(y_min, np.min(x[:, 1]))
        y_max = max(y_max, np.max(x[:, 1]))

    # Crear mapeo de índices
    if len(meshes) > 1:
        index = find_common_indices(all_coords[0], all_coords[1])
        total_nodes = sum(sizes) - len(index[0])
        
        # Mapear nodos
        node_mapping = {}
        current_index = 0
        
        # Primera malla
        for i in range(sizes[0]):
            node_mapping[i] = current_index
            current_index += 1
            
        # Segunda malla
        offset = sizes[0]
        for i in range(sizes[1]):
            if i in index[1]:
                # Usar índice de la primera malla para nodos compartidos
                idx = np.where(index[1] == i)[0][0]
                node_mapping[i + offset] = node_mapping[index[0][idx]]
            else:
                node_mapping[i + offset] = current_index
                current_index += 1
    else:
        total_nodes = sizes[0]
        node_mapping = {i: i for i in range(total_nodes)}

    def get_node_indices_all_meshes(condition):
        indices = set()
        for mesh_idx, mesh in enumerate(meshes):
            x = mesh.geometry.x
            mesh_indices = np.where(np.apply_along_axis(condition, 1, x))[0]
            if mesh_idx == 0:
                indices.update(mesh_indices)
            else:
                offset = sum(sizes[:mesh_idx])
                mapped_indices = [node_mapping[idx + offset] for idx in mesh_indices]
                indices.update(mapped_indices)
        return np.array(sorted(list(indices)))

    # Encontrar índices considerando nodos compartidos
    corner_down_left = get_node_indices_all_meshes(lambda coord: np.allclose(coord, [x_min, y_min, 0]))
    corner_down_rigth = get_node_indices_all_meshes(lambda coord: np.allclose(coord, [x_max, y_min, 0]))
    corner_up_rigth = get_node_indices_all_meshes(lambda coord: np.allclose(coord, [x_max, y_max, 0]))
    corner_up_left = get_node_indices_all_meshes(lambda coord: np.allclose(coord, [x_min, y_max, 0]))
    
    corner_indices = np.sort(np.hstack((corner_down_rigth, corner_up_rigth, corner_up_left)))

    def line_condition(coord, axis, t_d_l_r):
        line = np.array([[x_min, x_max],[y_min,y_max]])
        in_line = np.isclose(coord[1 - axis], line[1 - axis, t_d_l_r])
        less_max = coord[axis] < line[axis][1]
        over_min = coord[axis] > line[axis][0]
        return np.all([in_line, less_max, over_min])

    bottom_indices = get_node_indices_all_meshes(lambda coord: line_condition(coord, 0, 0))
    top_indices = get_node_indices_all_meshes(lambda coord: line_condition(coord, 0, 1))
    left_indices = get_node_indices_all_meshes(lambda coord: line_condition(coord, 1, 0))
    right_indices = get_node_indices_all_meshes(lambda coord: line_condition(coord, 1, 1))

    # Resto del código igual...
    num_nodes_x = total_nodes
    less_nodes = np.sort(np.hstack((corner_indices, top_indices, right_indices)))
    
    all_nodes = np.arange(num_nodes_x)
    reduced_nodes = np.setdiff1d(all_nodes, less_nodes)
    num_nodes_y = len(reduced_nodes)
    
    T = MatExtended().create()
    T.setSizes((2 * num_nodes_x, 2 * num_nodes_y))
    T.setType(PETSc.Mat.Type.AIJ)
    T.setUp()
    
    def set_matrix_values():
        # Relaciones de nodos de esquina
        for i in corner_indices:
            column = np.where(reduced_nodes == corner_down_left[0])[0]
            T.setValue(2*i, 2*column, 1.0)
            T.setValue(2*i+1, 2*column+1, 1.0)

        # Relaciones de bordes superior e inferior
        for i, j in zip(top_indices, bottom_indices):
            column = np.where(reduced_nodes == j)[0]
            T.setValue(2*i, 2*column, 1.0)
            T.setValue(2*i+1, 2*column+1, 1.0)

        # Relaciones de bordes derecho e izquierdo
        for i, j in zip(right_indices, left_indices):
            column = np.where(reduced_nodes == j)[0]
            T.setValue(2*i, 2*column, 1.0)
            T.setValue(2*i+1, 2*column+1, 1.0)

        # Nodos no restringidos
        for i in reduced_nodes:
            column = np.where(reduced_nodes == i)[0]
            T.setValue(2*i, 2*column, 1.0)
            T.setValue(2*i+1, 2*column+1, 1.0)

        T.assemblyBegin()
        T.assemblyEnd()
        return T

    T_matrix = set_matrix_values()

    def T_matrix_k(k_x, k_y):
        Lx = x_max - x_min
        Ly = y_max - y_min

        def change_values(index_row, index_columns, value):
            columns = np.where(np.isin(reduced_nodes, index_columns))[0].astype(np.int32)
            rows = index_row.astype(np.int32)
            for i, j in zip(rows, columns):
                T_matrix.setValue(2*i, 2*j, value)
                T_matrix.setValue(2*i+1, 2*j+1, value)

        change_values(right_indices, left_indices, np.exp(1j*Lx*k_x))
        change_values(top_indices, bottom_indices, np.exp(1j*Ly*k_y))
        change_values(corner_up_left, corner_down_left, np.exp(1j*Ly*k_y))
        change_values(corner_down_rigth, corner_down_left, np.exp(1j*Lx*k_x))
        change_values(corner_up_rigth, corner_down_left, np.exp(1j*(k_x*Lx + k_y*Ly)))
        
        T_matrix.assemblyBegin()
        T_matrix.assemblyEnd()
        
        return T_matrix

    return T_matrix_k