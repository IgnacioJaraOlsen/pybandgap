import pyvista
import numpy as np
from dolfinx.plot import vtk_mesh

pyvista.set_jupyter_backend("static")

def plot_IBZ(mesh, elements):
    p = pyvista.Plotter()

    tdim = mesh.topology.dim
    num_cells_local = mesh.topology.index_map(tdim).size_local
    marker = np.zeros((num_cells_local,3))
    marker[:,:]  = np.array([217, 67, 70])/255
    marker[elements,:] = np.array([67, 75, 217])/255
    mesh.topology.create_connectivity(tdim, tdim)
    topology, cell_types, x = vtk_mesh(mesh, tdim, np.arange(num_cells_local, dtype=np.int32))

    grid = pyvista.UnstructuredGrid(topology, cell_types, x).extract_surface(nonlinear_subdivision=0)
    grid.cell_data["colors"] = marker

    # Etiquetar los nodos
    node_coords = mesh.geometry.x  # Coordenadas de los nodos
    node_ids = np.arange(node_coords.shape[0])  # Identificadores de los nodos

    # Etiquetar las celdas
    cell_ids = np.arange(grid.n_cells)  # Obtener los identificadores de las celdas
    cell_centers = grid.cell_centers()

    font_size = 10

    p.add_point_labels(cell_centers, cell_ids, 
                            show_points=False, 
                            font_size=font_size, 
                            text_color="w", 
                            shape_opacity=0, 
                            justification_horizontal= 'center',
                            justification_vertical = 'center')

    p.add_point_labels(node_coords, node_ids, 
                       show_points=False, 
                       font_size=font_size, 
                       text_color="y",
                       shape_opacity=0, 
                       justification_horizontal='center',
                       justification_vertical='center')

    p.set_background('#1f1f1f')
    p.add_mesh(grid, show_edges=True, scalars="colors", rgb= True)
    p.add_axes(color='w')
    p.show_axes()
    p.view_xy()
    p.show()
    
def plot_structure(mesh, props,
                   colors = [
                       [67, 75, 217],
                       [217, 67, 70],
                       [97, 213, 157],
                       [213, 213, 97]
                       ]):
    
    p = pyvista.Plotter()

    tdim = mesh.topology.dim
    num_cells_local = mesh.topology.index_map(tdim).size_local

    has_diameters = hasattr(props, 'diameters')

    if has_diameters:
        line_widths = np.array([diameter for diameter in props.diameters.values()])        
        line_widths = np.exp(line_widths* 250)

    # Set color and line width based on materials and diameters
    marker = np.array([np.array(colors[material.creation_number])/255 for material in props.materials.values()])
    
    mesh.topology.create_connectivity(tdim, tdim)
    topology, cell_types, x = vtk_mesh(mesh, tdim, np.arange(num_cells_local, dtype=np.int32))

    grid = pyvista.UnstructuredGrid(topology, cell_types, x).extract_surface(nonlinear_subdivision=0)
    grid.cell_data["colors"] = marker

    p.set_background('#1f1f1f')
    #p.add_mesh(grid, show_edges=True, scalars="colors", rgb=True)
    p.add_axes(color='w')

    # Set line widths for each cell
    for cell_id in range(num_cells_local):
        if has_diameters:
            p.add_mesh(grid.extract_cells(cell_id), 
                       color=marker[cell_id], 
                       line_width=line_widths[cell_id])
        else:
            p.add_mesh(grid.extract_cells(cell_id), color=marker[cell_id])
            

    p.show_axes()
    p.view_xy()
    p.show()
