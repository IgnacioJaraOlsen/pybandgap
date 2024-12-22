import numpy as np
from petsc4py import PETSc
from pybandgap.mass_and_stiffness_matrix import matrix_and_stiffness_matrix


def find_common_indices(array1, array2):
    array1_rows = {tuple(row): idx for idx, row in enumerate(array1)}
    
    # Lists to store indices
    indices_array1 = []
    indices_array2 = []
    
    # Search for common rows
    for idx2, row in enumerate(array2):
        row_tuple = tuple(row)
        if row_tuple in array1_rows:
            indices_array1.append(array1_rows[row_tuple])
            indices_array2.append(idx2)
    
    return np.vstack(np.array(indices_array1), np.array(indices_array2))

def mix_matrix(meshes, props):
    mass_matrices = []
    stiffness_matrices = []
    coords = []
    sizes = []
    
    # Obtener matrices y tamaños individuales
    for mesh, prop in zip(meshes, props):
        mass, stiffness = matrix_and_stiffness_matrix(mesh, prop)
        mass_matrices.append(mass)
        stiffness_matrices.append(stiffness)
        coords.append(mesh.geometry.x)
        sizes.append(mass.shape[0])
    
    # Obtener índices comunes y expandirlos para los DOFs
    index = find_common_indices(*coords)
    index_dofs = np.hstack((index*2, index*2+1))
    
    # Calcular tamaño total
    total_size = sum(sizes) - len(index_dofs)
    
    # Crear matrices PETSc globales
    M_global = PETSc.Mat().createAIJ([total_size, total_size])
    K_global = PETSc.Mat().createAIJ([total_size, total_size])
    
    # Preallocate memoria (esto mejora el rendimiento)
    M_global.setUp()
    K_global.setUp()
    
    # Crear mapeo de índices
    shared_dof_map = {}
    for i, j in zip(index_dofs[0], index_dofs[1]):
        shared_dof_map[j] = i
    
    # Insertar primera matriz
    for i in range(sizes[0]):
        for j in range(sizes[0]):
            M_global.setValues(i, j, mass_matrices[0][i, j])
            K_global.setValues(i, j, stiffness_matrices[0][i, j])
    
    # Insertar segunda matriz
    offset = sizes[0] - len(index_dofs[0])
    for i in range(sizes[1]):
        for j in range(sizes[1]):
            # Determinar índices en la matriz global
            row = shared_dof_map[i] if i in shared_dof_map else offset + i
            col = shared_dof_map[j] if j in shared_dof_map else offset + j
            
            # Añadir valores a las matrices globales
            M_global.setValues(row, col, mass_matrices[1][i, j], addv=True)
            K_global.setValues(row, col, stiffness_matrices[1][i, j], addv=True)
    
    # Ensamblar las matrices finales
    M_global.assemblyBegin()
    M_global.assemblyEnd()
    K_global.assemblyBegin()
    K_global.assemblyEnd()
    
    return M_global, K_global