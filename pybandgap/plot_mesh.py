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

    font_size = 15

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
    
def plot_IBZ_with_materials_and_diameters(mesh, materials_dict, diameters_dict):
    p = pyvista.Plotter()

    tdim = mesh.topology.dim
    num_cells_local = mesh.topology.index_map(tdim).size_local
    marker = np.zeros((num_cells_local, 3))
    line_widths = np.ones(num_cells_local)  # Default line width
    
    # Set color and line width based on materials and diameters
    for cell_id, material in materials_dict.items():
        if material.creation_number == 1:
            marker[cell_id, :] = np.array([67, 75, 217])/255  # Color for material 1
        elif material.creation_number == 2:
            marker[cell_id, :] = np.array([217, 67, 70])/255  # Color for material 2

    for cell_id, diameter in diameters_dict.items():
        line_widths[cell_id] = diameter*10  # Set line width based on diameter
        
    line_widths = np.exp(line_widths* 25)

    mesh.topology.create_connectivity(tdim, tdim)
    topology, cell_types, x = vtk_mesh(mesh, tdim, np.arange(num_cells_local, dtype=np.int32))

    grid = pyvista.UnstructuredGrid(topology, cell_types, x).extract_surface(nonlinear_subdivision=0)
    grid.cell_data["colors"] = marker

    cell_ids = np.arange(grid.n_cells)  # Obtener los identificadores de las celdas
    cell_centers = grid.cell_centers()

    # p.add_point_labels(cell_centers, cell_ids, 
    #                         show_points=False, 
    #                         font_size=10, 
    #                         text_color="w", 
    #                         shape_opacity=0, 
    #                         justification_horizontal='center',
    #                         justification_vertical='center')

    p.set_background('#1f1f1f')
    #p.add_mesh(grid, show_edges=True, scalars="colors", rgb=True)
    p.add_axes(color='w')

    # Set line widths for each cell
    for cell_id in range(num_cells_local):
        p.add_mesh(grid.extract_cells(cell_id), 
                   color=marker[cell_id], 
                   line_width=line_widths[cell_id], 
                   )

    p.show_axes()
    p.view_xy()
    p.show()
