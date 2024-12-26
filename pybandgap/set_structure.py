import numpy as np
from dataclasses import dataclass
from dolfinx.mesh import compute_midpoints

class Material:
    # Variable de clase para contar el número de instancias
    counter = -1

    def __init__(self, material, young_modulus, poisson_ratio, density):
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.density = density
        self.material = material

        # Incrementar el contador cada vez que se crea una nueva instancia
        Material.counter += 1
        # Asignar el número de creación a la instancia
        self.creation_number = Material.counter

    def __repr__(self):
        return f'{self.material} (Creation Number: {self.creation_number})'

    @classmethod
    def get_counter(cls):
        return int(cls.counter)

class Props:
    def __init__(self, mesh):
        self.mesh = mesh
    
    def set_prop(self, name, prop):
        prop = map_prop(self.mesh, prop)
        setattr(self, name, prop)

def get_midpoint_elements(mesh):
    tdim = mesh.topology.dim
    mesh_entities = mesh.topology.index_map(tdim).size_local
    entities = np.arange(mesh_entities)
    
    midpoints = compute_midpoints(mesh, tdim, entities)
    return midpoints

def get_midpoint_mesh(mesh):
    
    x = mesh.geometry.x
    x_min = np.min(x[:, 0])
    x_max = np.max(x[:, 0])
    y_min = np.min(x[:, 1])
    y_max = np.max(x[:, 1])
    z_min = np.min(x[:, 2])
    z_max = np.max(x[:, 2])
    
    midpoint_mesh = [(x_max+x_min)/2, (y_max + y_min)/2, (z_max + z_min)/2]
    return midpoint_mesh

def fine_irreducible_brillouin_zone(mesh):    
    
    midpoint_mesh = get_midpoint_mesh(mesh)
    midpoints = get_midpoint_elements(mesh)

    x_limit = (midpoints[:, 0] >= midpoint_mesh[0])
    y_limit = (midpoints[:, 1] >= midpoint_mesh[1])
    xy_limit = ((midpoints[:, 1] < midpoints[:, 0]) | 
                np.isclose(midpoints[:, 1], midpoints[:, 0]))
    
    conditon =  np.vstack(
        (x_limit, y_limit, xy_limit)
    )
    
    indices = np.where(np.all(conditon, axis=0))[0]
    
    return indices, conditon

def symmetric_projection(center, point, axis = 0):
    axis = 1 - axis
    punto_simetrico = point.copy()
    
    distance = center - point
    
    punto_simetrico[axis] +=  2*distance[axis]
    
    return punto_simetrico


def symmetric_projection_diagonal(p1, p2, axis = None):
    # Transformar p2 al nuevo sistema de coordenadas (restar el origen)
    punto_relativo = p2 - p1
    
    # Calcular coordenadas relativas
    x, y = punto_relativo[0], punto_relativo[1]
    
    # Determinar el eje de simetría
    if axis is None:
        # Elegir automáticamente la línea más cercana
        dist_y_igual_x = abs(y - x)
        dist_y_igual_menos_x = abs(y + x)
        axis = 1 if dist_y_igual_x <= dist_y_igual_menos_x else -1
    
    # Realizar simetría según el eje seleccionado
    if axis == 1:
        # Simetría respecto a y = x (diagonal positiva)
        punto_simetrico_relativo = np.array([
            y,      # x cambia a y
            x,      # y cambia a x
            punto_relativo[2]  # z permanece igual
        ])
    elif axis == -1:
        # Simetría respecto a y = -x (diagonal negativa)
        punto_simetrico_relativo = np.array([
            -y,     # x cambia a -y
            -x,     # y cambia a -x
            punto_relativo[2]  # z permanece igual
        ])
    else:
        raise ValueError("El parámetro axis solo puede ser None, 1 o -1")
    
    # Regresar al sistema de coordenadas original sumando el nuevo origen
    punto_simetrico = punto_simetrico_relativo + p1
    
    return punto_simetrico, axis

def apply_symmetric_projection(center, points, axis, direction = None):
    projections = []
    for point in points:
        if direction == 'd':
            projection, _ = symmetric_projection_diagonal(center, point, axis)
        else:
            projection = symmetric_projection(center, point, axis)
            
        projections.append(projection)
    return np.array(projections)

def get_nodes_element(mesh, index):
    points = mesh.geometry.x[mesh.topology.connectivity(mesh.topology.dim, 0).links(index)]
    return points

def rows_in_array(arr1, arr2, tolerance=1e-8):
    return np.array([
        np.any(np.all(np.isclose(fila, arr2, atol=tolerance, rtol=0), axis=1)) 
        for fila in arr1
    ])
    
def in_diagonal(point, midpoint_mesh):
    punto_relativo = point - midpoint_mesh
    x, y = punto_relativo[0], punto_relativo[1]
    
    return np.isclose(x, y, atol=1e-10) | np.isclose(-x, y, atol=1e-10)
    
def fine_symmetry_elements(mesh, index):
    
    midpoint_mesh = get_midpoint_mesh(mesh)
    
    midpoints = get_midpoint_elements(mesh)
    
    point_index  = midpoints[index,:]
    
    def get_index(i, point, axis, direction = None):
        
        index_projection = np.where(np.all(np.isclose(point, midpoints, atol=1e-10), axis=1))[0]

        if len(index_projection) > 1:            
            nodes_index = get_nodes_element(mesh, i)
            projection = apply_symmetric_projection(midpoint_mesh, nodes_index, axis, direction = direction)
            
            nodes_projection_1 = get_nodes_element(mesh, index_projection[0])
            
            if rows_in_array(projection, nodes_projection_1).all():
                return index_projection[0]
            else:
                return index_projection[1]
        
        return index_projection
    
    index_array = np.array([index])
    
    if ~in_diagonal(point_index, midpoint_mesh):
    
        diagonal_symmetry_point, axis = symmetric_projection_diagonal(midpoint_mesh, point_index)
        if diagonal_symmetry_point is not None:
            index_array = np.append(index_array, get_index(index, diagonal_symmetry_point, axis, direction = 'd'))
        
    x_simetry = np.array([])
    for i in index_array:
        x_symmetry_point = symmetric_projection(midpoint_mesh, midpoints[i,:], axis= 0)
        x_simetry = np.append(x_simetry, get_index(i, x_symmetry_point, 0))
    
    index_array = np.append(index_array, x_simetry).astype(int)

    y_simetry = np.array([])
    for i in index_array:
        y_symmetry_point = symmetric_projection(midpoint_mesh, midpoints[i,:], axis= 1)
        y_simetry = np.append(y_simetry, get_index(i, y_symmetry_point, 1))
    
    index_array = np.append(index_array, y_simetry).astype(int)
    
    return index_array

def map_prop(mesh, prop):
    elements_in_ibz, _ = fine_irreducible_brillouin_zone(mesh)
    
    prop_map = {
        i: m for i, m in zip(elements_in_ibz , prop)
        }
    
    map_elements = map(lambda index: fine_symmetry_elements(mesh, index), 
                       elements_in_ibz)
    
    for i, simetry in zip(elements_in_ibz, map_elements):
        prop_map.update(dict.fromkeys(simetry, prop_map[i]))
    
    return dict(sorted(prop_map.items()))