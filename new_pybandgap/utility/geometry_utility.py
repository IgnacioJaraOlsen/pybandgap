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
    
    structure.boundary_conditions = boundary_conditions
    structure.reference_nodes = reference_nodes


def graham_scan(points):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    if len(points) < 3:
        return points

    points = np.array(points)
    bottom_point = points[np.lexsort((points[:,0], points[:,1]))][0]
    
    sorted_points = sorted(
        [tuple(point) for point in points],
        key=lambda p: (
            np.arctan2(p[1] - bottom_point[1], p[0] - bottom_point[0]),
            (p[0] - bottom_point[0])**2 + (p[1] - bottom_point[1])**2
        )
    )
    
    stack = [sorted_points[0], sorted_points[1]]
    
    for i in range(2, len(sorted_points)):
        while len(stack) > 1 and orientation(stack[-2], stack[-1], sorted_points[i]) != 2:
            stack.pop()
        stack.append(sorted_points[i])
    
    return np.array(stack)

def find_intersections_with_line(points, center, angle, tolerance=1e-6):
    def line_intersection(p1, p2, center, angle):
        cx, cy = center[0], center[1]
        dx, dy = np.cos(angle), np.sin(angle)
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]

        A = np.array([[dx, x1 - x2], [dy, y1 - y2]])
        b = np.array([x1 - cx, y1 - cy])

        try:
            t, s = np.linalg.solve(A, b)
            if 0 <= s <= 1:
                return np.array([x1 + s * (x2 - x1), y1 + s * (y2 - y1), 0])
        except np.linalg.LinAlgError:
            pass
        return None

    def is_existing_point(point, points, tolerance):
        for p in points:
            if np.linalg.norm(point[:2] - p[:2]) < tolerance:
                return True
        return False

    def reorder_points(points, center):
        distances = [np.linalg.norm(point - center) for point in points]
        min_index = np.argmin(distances)
        return np.roll(points, -min_index, axis=0)

    new_points = []
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]

        new_points.append(p1)

        intersection = line_intersection(p1, p2, center, angle)
        if intersection is not None and not is_existing_point(intersection, points, tolerance):
            new_points.append(intersection)

    new_points = reorder_points(np.array(new_points), center)
    return new_points


def calculate_slope(points):
    point1, point2 = points
    
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    
    # Avoid division by zero
    if np.isclose(x2 - x1, 0):
        return float('inf')  # The slope is infinite (vertical line)
    
    slope = (y2 - y1) / (x2 - x1)
    return slope
