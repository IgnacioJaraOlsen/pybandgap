import numpy as np
from petsc4py import PETSc
from pybandgap.mass_and_stiffness_matrix import mass_and_stiffness_matrix


def find_common_indices(array1, array2, tol=1e-10):
    # Convertir filas a tuplas para su uso en un diccionario, con redondeo para la tolerancia
    array1_rows = {tuple(np.round(row, decimals=int(-np.log10(tol)))): idx for idx, row in enumerate(array1)}
    
    # Listas para almacenar los índices
    indices_array1 = []
    indices_array2 = []
    
    # Buscar filas comunes considerando la tolerancia
    for idx2, row in enumerate(array2):
        row_tuple = tuple(np.round(row, decimals=int(-np.log10(tol))))
        if row_tuple in array1_rows:
            # Verificar si las filas son efectivamente similares dentro de la tolerancia
            if np.allclose(array1[array1_rows[row_tuple]], row, atol=tol):
                indices_array1.append(array1_rows[row_tuple])
                indices_array2.append(idx2)
    
    return np.vstack((np.array(indices_array1), np.array(indices_array2)))

def mix_matrix(meshes, props):
    mass_matrices = []
    stiffness_matrices = []
    coords = []
    sizes = []

    for mesh, prop in zip(meshes, props):
        mass, stiffness = mass_and_stiffness_matrix(mesh, prop)
        mass_matrices.append(mass)
        stiffness_matrices.append(stiffness)
        coords.append(mesh.geometry.x)
        sizes.append(mass.getSize()[0])

    index = find_common_indices(*coords)
    # Convertir índices de nodos a índices de DOFs
    index_dofs = np.vstack((
        np.hstack((index[0] * 2, index[0] * 2 + 1)),  # DOFs para la primera malla
        np.hstack((index[1] * 2, index[1] * 2 + 1))   # DOFs para la segunda malla
    )).astype(np.int32)  # Convertir a int32

    # numero de indices totales de las mallas combinadas
    total_size = sum(sizes) - len(index_dofs[0])

    # Crear matrices globales con preallocación
    M_global = PETSc.Mat().createAIJ([total_size, total_size])
    K_global = PETSc.Mat().createAIJ([total_size, total_size])
    
    # Preallocación: estimar número de elementos no cero por fila
    nnz = int(total_size * 0.1)  # estimación del 10% de elementos no cero
    M_global.setPreallocationNNZ(nnz)
    K_global.setPreallocationNNZ(nnz)

    M_global.zeroEntries()
    K_global.zeroEntries()

    # Inicializar el contador de offset para la asignación de índices
    offset = 0
    
    # Para cada malla, insertar sus matrices locales en las globales
    for i, (mass, stiff) in enumerate(zip(mass_matrices, stiffness_matrices)):
        size = sizes[i]  # Ya incluye los DOFs
        
        # Crear índices locales y globales para esta malla, asegurando tipo int32
        local_indices = np.arange(size, dtype=np.int32)
        global_indices = (local_indices + offset).astype(np.int32)
        
        # Si no es la primera malla, ajustar los índices compartidos y posteriores
        if i > 0:
            # Ordenar los índices para procesarlos en orden
            sorted_pairs = sorted(zip(index_dofs[1], index_dofs[0]), key=lambda x: x[0])
            
            # Contador para llevar el ajuste acumulativo
            cumulative_shift = 0
            
            for local_idx, shared_idx in sorted_pairs:
                # Ajustar el índice actual
                global_indices[local_idx] = shared_idx
                
                # Decrementar todos los índices posteriores
                mask = local_indices > local_idx
                global_indices[mask] -= 1
                cumulative_shift += 1
        
        
        # Transferir valores de las matrices locales a las globales
        for row in range(size):
            # Obtener los valores e índices de columna para esta fila
            cols, vals = mass.getRow(row)
            M_global.setValues(
                global_indices[row], 
                global_indices[cols], 
                vals,
                addv = True
            )
            
            cols, vals = stiff.getRow(row)
            K_global.setValues(
                global_indices[row], 
                global_indices[cols], 
                vals,
                addv = True
            )
        
        # Actualizar el offset para la siguiente malla
        offset = size

    # Ensamblar las matrices finales
    M_global.assemblyBegin()
    M_global.assemblyEnd()
    K_global.assemblyBegin()
    K_global.assemblyEnd()

    return M_global, K_global