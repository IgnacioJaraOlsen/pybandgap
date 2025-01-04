import numpy as np

def calculate_symmetric_point(P1, P2, angle_xy, threshold=1e-10):
    """
    Find the symmetric point of P1 with respect to the line passing through P2 and with angle angle_xy.
    """
    # Dirección de la línea con el ángulo dado
    direction = np.array([np.cos(angle_xy), np.sin(angle_xy), 0])
    direction = direction / np.linalg.norm(direction)  # Asegurarse de que la dirección sea una unidad

    # Vector de P2 a P1
    vector_p2p1 = P1 - P2
    # Longitud de la proyección sobre la dirección de la línea
    projection_length = np.dot(vector_p2p1, direction)
    
    # Punto proyectado sobre la línea
    projection_point = P2 + projection_length * direction
    
    # Punto simétrico (espejo) respecto a la línea
    symmetric_point = 2 * projection_point - P1

    # Redondear valores muy pequeños a 0
    symmetric_point[np.abs(symmetric_point) < threshold] = 0
    return symmetric_point

def quadrilateral_perimeter(structure):
    '''
    up_left          top        up_right
            +-------------------+
            |                   |
            |                   |
      left  |                   | right
            |                   |
            |                   |
            +-------------------+
    down_left      bottom       down_right
    '''
    boundary_conditions = {
        'down_left':  lambda x: np.isclose(x[:, 0], structure.x_min) & np.isclose(x[:, 1], structure.y_min),
        'down_right': lambda x: np.isclose(x[:, 0], structure.x_max) & np.isclose(x[:, 1], structure.y_min),
        'up_left':    lambda x: np.isclose(x[:, 0], structure.x_min) & np.isclose(x[:, 1], structure.y_max),
        'up_right':   lambda x: np.isclose(x[:, 0], structure.x_max) & np.isclose(x[:, 1], structure.y_max),
        'bottom': lambda x: np.isclose(x[:, 1], structure.y_min) & (structure.x_min < x[:, 0]) & (x[:, 0] < structure.x_max),
        'top':    lambda x: np.isclose(x[:, 1], structure.y_max) & (structure.x_min < x[:, 0]) & (x[:, 0] < structure.x_max),
        'left':   lambda x: np.isclose(x[:, 0], structure.x_min) & (structure.y_min < x[:, 1]) & (x[:, 1] < structure.y_max),
        'right':  lambda x: np.isclose(x[:, 0], structure.x_max) & (structure.y_min < x[:, 1]) & (x[:, 1] < structure.y_max)
    }
    
    reference_nodes = {
        'down_left': ['down_right', 'up_left', 'up_right'],
        'bottom': ['top'],
        'left': ['right'],
    }
    
    return boundary_conditions, reference_nodes