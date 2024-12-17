import numpy as np
from dolfinx.mesh import compute_midpoints

def custom_locate_entities(mesh, condition_func):
    tdim = mesh.topology.dim
    # Obtener todos los elementos de la dimensión especificada
    mesh_entities = mesh.topology.index_map(tdim).size_local
    
    # Crear un array de índices de entidades
    entities = np.arange(mesh_entities)
    
    # Obtener los midpoints de los elementos
    midpoints = compute_midpoints(mesh, tdim, entities)
    
    # Encontrar los índices de los elementos que cumplen la condición
    indices = np.where(condition_func(midpoints, mesh))[0]
    
    return indices

def in_irreducible_brillouin_zone(centers, mesh):
    x = mesh.geometry.x
    x_min = np.min(x[:, 0])
    x_max = np.max(x[:, 0])
    y_min = np.min(x[:, 1])
    y_max = np.max(x[:, 1])
    
    x_limit = centers[:, 0] <= (x_max - x_min)/2
    y_limit = centers[:, 1] <= (y_max - y_min)/2
    diagonal = (centers[:, 1] < centers[:, 0]) | np.isclose(centers[:, 1], centers[:, 0])
    
    return x_limit & y_limit & diagonal

def custom_locate_symmetry(index, mesh, condition_func):
    f_lambda  = lambda centers, mesh: condition_func(index, centers, mesh)
    indices = custom_locate_entities(mesh, f_lambda)
    return indices

def get_node_slope(mesh, index):
    points = mesh.geometry.x[mesh.topology.connectivity(mesh.topology.dim, 0).links(index)]
    point1 = points[0]
    point2 = points[1]
    x1, y1, _ = point1
    x2, y2, _ = point2    
    return (y2 - y1) / (x2 - x1)

def symmetry_condition(index, centers, mesh):
    x = mesh.geometry.x
    x_max = np.max(x[:, 0])
    x_min = np.min(x[:, 0])
    y_max = np.max(x[:, 1])  
    y_min = np.min(x[:, 0])  
    
    point_index  = centers[index,:]
    x_symetry = lambda point: [point[0], y_max - point[1], 0]
    y_symetry = lambda point: [x_max - point[0], point[1], 0]
    
    rigth_diagonal_point = [point_index[1], point_index[0], 0]
    x_symmetry_point_1 = x_symetry(point_index)
    x_symmetry_point_2 = x_symetry(rigth_diagonal_point)
    y_symmetry_point_1 = y_symetry(point_index)
    y_symmetry_point_2 = y_symetry(rigth_diagonal_point)
    y_symmetry_point_3 = y_symetry(x_symmetry_point_1)
    y_symmetry_point_4 = y_symetry(x_symmetry_point_2)
    
    condition = np.all(np.isclose(centers, rigth_diagonal_point, atol=1e-3), axis=1) & ~np.all(np.isclose(point_index, rigth_diagonal_point, atol=1e-3))
    condition = np.all(np.isclose(centers, x_symmetry_point_1, atol=1e-3), axis=1) | condition # x_symmetry_1
    condition = np.all(np.isclose(centers, x_symmetry_point_2, atol=1e-3), axis=1) | condition # x_symmetry_2
    condition = np.all(np.isclose(centers, y_symmetry_point_1, atol=1e-3), axis=1) | condition # y_symmetry_1
    condition = np.all(np.isclose(centers, y_symmetry_point_2, atol=1e-3), axis=1) | condition # y_symmetry_2
    condition = np.all(np.isclose(centers, y_symmetry_point_3, atol=1e-3), axis=1) | condition # y_symmetry_3
    condition = np.all(np.isclose(centers, y_symmetry_point_4, atol=1e-3), axis=1) | condition # y_symmetry_4

    vectorized_slope = np.vectorize(lambda index: get_node_slope(mesh, index))

    # Calculamos las pendientes para todos los índices
    slopes = vectorized_slope(range(len(centers)))

    return condition


def set_materia(mesh, materials):
    elements_in_ibz = custom_locate_entities(mesh, in_irreducible_brillouin_zone)
    
    materials_map = {
        i: m for i, m in zip(elements_in_ibz , materials)
        }
    
    map_elements = map(lambda x: custom_locate_symmetry(x, mesh, symmetry_condition), 
                       elements_in_ibz)
    
    for i, simetry in zip(elements_in_ibz, map_elements):
            materials_map.update(dict.fromkeys(simetry, materials_map[i]))
    
    return materials_map