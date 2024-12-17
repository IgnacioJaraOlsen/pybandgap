import gmsh
from dolfinx import io
from mpi4py import MPI

def truss_like_cross():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Desactivar la información de malla
    geom = gmsh.model.geo

    Lb = 2.5e-2  # Longitud de la barra
    nc = 4  # Número de divisiones en cada dirección

    points = []
    index = 0
    for i in range(nc + 1):
        for e in range(nc + 1):
            # Crear puntos de izquierda a derecha y abajo hacia arriba
            points.append(geom.add_point(Lb * e, Lb * i, 0.0, tag = index))
            index +=  1
        index +=  1
    lines = []

    # Crear líneas horizontales
    for i in range(nc + 1):
        points_horizontal_lines = points[i * (nc + 1):(i + 1) * (nc + 1)]
        lines += [geom.add_line(points_horizontal_lines[e], points_horizontal_lines[e + 1]) for e in range(nc)]

    # Crear líneas verticales
    for e in range(nc + 1):
        points_vertical_lines = points[e::(nc + 1)]  # Desplazar los índices para las líneas verticales
        lines += [geom.add_line(points_vertical_lines[i], points_vertical_lines[i + 1]) for i in range(nc)]

    # Crear líneas diagonales
    for i in range(nc):
        points_diagonals_1 = points[i * (nc + 1):(i + 1) * (nc + 1)]
        points_diagonals_2 = points[(i + 1) * (nc + 1):(i + 2) * (nc + 1)]
        lines += [geom.add_line(points_diagonals_1[e], points_diagonals_2[e + 1]) for e in range(nc)]
        lines += [geom.add_line(points_diagonals_1[e + 1], points_diagonals_2[e]) for e in range(nc)]

    geom.synchronize()

    # Asignar líneas transfinita (suavizado de la malla) para las líneas
    for l in lines:
        gmsh.model.mesh.set_transfinite_curve(l, 2)

    gmsh.model.add_physical_group(1, lines, 2)

    gmsh.model.mesh.generate(dim=1)

    domain, markers, facets = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)

    gmsh.finalize()

    return domain, markers, facets