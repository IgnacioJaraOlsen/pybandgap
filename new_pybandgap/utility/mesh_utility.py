import numpy as np
from dolfinx.mesh import compute_midpoints

def get_midpoints(mesh):
    tdim = mesh.topology.dim
    mesh_entities = mesh.topology.index_map(tdim).size_local
    return compute_midpoints(mesh, tdim, np.arange(mesh_entities))

def get_nodes_element(mesh, index):
        return mesh.geometry.x[mesh.topology.connectivity(mesh.topology.dim, 0).links(index)]
    
