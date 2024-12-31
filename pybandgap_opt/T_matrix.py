import numpy as np
from petsc4py import PETSc

class MatExtended(PETSc.Mat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def conjugate_transpose(self):
        mat_transpose = self.transpose()
        mat_conjugate_transpose = mat_transpose.conjugate()
        return mat_conjugate_transpose

def T_matrix(structure):
    """
    Creates T matrix using SetStructure instance instead of raw meshes.
    
    Args:
        structure: SetStructure instance containing mesh information
    """
    # Get dimensions from structure
    x_min = structure.x_min
    x_max = structure.x_max
    y_min = structure.y_min
    y_max = structure.y_max
    
    # Get node indices from structure
    corner_down_left = structure.node_down_left
    corner_down_right = structure.node_down_right
    corner_up_right = structure.node_up_right
    corner_up_left = structure.node_up_left
    
    bottom_indices = structure.node_bottom
    top_indices = structure.node_top
    left_indices = structure.node_left
    right_indices = structure.node_right
    
    # Combine corner indices
    corner_indices = np.sort(np.hstack((corner_down_right, corner_up_right, corner_up_left)))
    
    # Get total nodes and reduced nodes from structure
    num_nodes_x = structure.total_nodes
    less_nodes = np.sort(np.hstack((corner_indices, top_indices, right_indices)))
    
    all_nodes = np.arange(num_nodes_x)
    reduced_nodes = np.setdiff1d(all_nodes, less_nodes)
    num_nodes_y = len(reduced_nodes)
    
    # Create T matrix
    T = MatExtended().create()
    T.setSizes((2 * num_nodes_x, 2 * num_nodes_y))
    T.setType(PETSc.Mat.Type.AIJ)
    T.setUp()
    
    def set_matrix_values():
        # Corner nodes
        for i in corner_indices:
            column = np.where(reduced_nodes == corner_down_left[0])[0]
            T.setValue(2*i, 2*column, 1.0)
            T.setValue(2*i+1, 2*column+1, 1.0)

        # Top and bottom edges
        for i, j in zip(top_indices, bottom_indices):
            column = np.where(reduced_nodes == j)[0]
            T.setValue(2*i, 2*column, 1.0)
            T.setValue(2*i+1, 2*column+1, 1.0)

        # Right and left edges
        for i, j in zip(right_indices, left_indices):
            column = np.where(reduced_nodes == j)[0]
            T.setValue(2*i, 2*column, 1.0)
            T.setValue(2*i+1, 2*column+1, 1.0)

        # Interior nodes
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
        change_values(corner_down_right, corner_down_left, np.exp(1j*Lx*k_x))
        change_values(corner_up_right, corner_down_left, np.exp(1j*(k_x*Lx + k_y*Ly)))
        
        T_matrix.assemblyBegin()
        T_matrix.assemblyEnd()
        
        return T_matrix
    
    return T_matrix_k