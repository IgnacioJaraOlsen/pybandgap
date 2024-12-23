import sys
import gmsh
import numpy as np
from dolfinx import io
from mpi4py import MPI

def truss_part(ne_x, ne_y, Lx, Ly):
    nx = ne_x + 1
    ny = ne_y + 1
    
    x_linspace = np.linspace(-Lx/2, Lx/2, nx)
    y_linspace = np.linspace(-Ly/2, Ly/2, ny)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Desactivar la información de malla
    geom = gmsh.model.geo
    
    points = [geom.add_point(x,y,0) for y in y_linspace for x in x_linspace]
    
    lines = [geom.add_line(points[i], points[i+1]) for i in range(0, len(points)-1, 1) if (i+1) % nx != 0]
    
    lines += [geom.add_line(points[i], points[i+nx]) for i in range(0, len(points)-nx, 1)]
    
    lines += [geom.add_line(points[i], points[i+nx+1]) for i in range(0, len(points)-nx-1, 1) if (i+1) % nx != 0]
    
    lines += [geom.add_line(points[i], points[i+nx-1]) for i in range(0, len(points)-nx, 1) if i % nx != 0]
        
    gmsh.model.geo.synchronize()

    gmsh.model.add_physical_group(0, points, 1)
    
    gmsh.model.add_physical_group(1, lines, 2)

    # Definir el área de interés (por ejemplo, un cuadrado de (0.5, 0.5) a (1.5, 1.5))
    xmin, xmax = -Lx/4, Lx/4
    ymin, ymax = -Ly/4, Lx/4

    # Identificar puntos dentro del área de interés
    points_to_remove = []
    for point in gmsh.model.getEntities(0):
        point_tag = point[1]
        coords = gmsh.model.getValue(0, point_tag, [])
        x, y, z = coords[0], coords[1], coords[2]
        if xmin <= x <= xmax and ymin <= y <= ymax:
            points_to_remove.append(point_tag)

    # Identificar líneas que tengan ambos puntos dentro del área de interés
    lines_to_remove = []
    for line in gmsh.model.getEntities(1):
        line_tag = line[1]
        line_points = gmsh.model.getAdjacencies(1, line_tag)[1]
        if all(point in points_to_remove for point in line_points):
            lines_to_remove.append(line_tag)

    # Eliminar las líneas identificadas
    for line_tag in lines_to_remove:
        gmsh.model.geo.remove([(1, line_tag)])

    # Eliminar los puntos identificados (incluido el punto central)
    for point_tag in points_to_remove:
        gmsh.model.geo.remove([(0, point_tag)])

    # Sincronizar para aplicar los cambios
    gmsh.model.geo.synchronize()

    for line in gmsh.model.getEntities(1):
        line_tag = line[1]
        gmsh.model.mesh.set_transfinite_curve(line_tag, 2)

    gmsh.model.mesh.generate(dim=1)
        
    domain, markers, facets = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    
    gmsh.finalize()
    
    return domain, markers, facets
    
def plate_part(ne_x, ne_y, Lx, Ly):
    nx = ne_x + 1
    ny = ne_y + 1    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # Desactivar la información de malla
    geom = gmsh.model.geo

    x_linspace = np.linspace(-Lx/2, Lx/2, nx)
    y_linspace = np.linspace(-Ly/2, Ly/2, ny)

    order = [(x, y_linspace[0]) for x in x_linspace]
    order += [(x_linspace[-1],y) for y in y_linspace[1:]]
    order += [(x,y_linspace[-1]) for x in x_linspace[:-1][::-1]]
    order += [(x_linspace[0],y) for y in y_linspace[1:-1][::-1]]
    
    points = [geom.add_point(x,y,0) for x,y in order]
    
    lines = [geom.add_line(points[i], points[i+1]) for i in range(0, len(points)-1, 1)]
    lines += [geom.add_line(points[-1], points[0])]
    
    loop = geom.addCurveLoop(lines)
    
    surface = geom.addPlaneSurface([loop])
    
    gmsh.model.geo.synchronize()
    
    gmsh.model.add_physical_group(0, points, 1)
    
    gmsh.model.add_physical_group(1, lines, 2)
    
    gmsh.model.add_physical_group(2, [surface], 3)
    
    geom.mesh.setTransfiniteSurface(surface, cornerTags= [points[i] for i in range(0, len(points), 2)])
    geom.mesh.setRecombine(2, surface)
    
    gmsh.model.geo.synchronize() 
     
    gmsh.model.mesh.generate(dim=2)
    
    domain, markers, facets = io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    
    return domain, markers, facets  


if __name__ == "__main__":
    Lc = 1
    ne = 4
    domain, markers, facets = truss_part(ne, ne, Lc, Lc)
    domain2, markers2, facets2 = plate_part(ne-2, ne-2, Lc/2, Lc/2)
    
    import pyvista
    from dolfinx.plot import vtk_mesh

    p = pyvista.Plotter()
    tdim = domain.topology.dim
    num_cells_local = domain.topology.index_map(tdim).size_local
    topology, cell_types, x = vtk_mesh(domain, tdim, np.arange(num_cells_local, dtype=np.int32))

    tdim2 = domain2.topology.dim
    num_cells_local2 = domain2.topology.index_map(tdim2).size_local
    topology2, cell_types2, x2 = vtk_mesh(domain2, tdim2, np.arange(num_cells_local2, dtype=np.int32))

    grid = pyvista.UnstructuredGrid(topology, cell_types, x).extract_surface(nonlinear_subdivision=0)
    grid2 = pyvista.UnstructuredGrid(topology2, cell_types2, x2).extract_surface(nonlinear_subdivision=0)
    
    p.add_mesh(grid, show_edges=True)
    p.add_mesh(grid2, show_edges=True)
    
    p.add_axes(color='w')
    p.set_background('#1f1f1f')
    p.view_xy()
    p.show()