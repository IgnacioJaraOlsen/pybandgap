import numpy as np
from petsc4py import PETSc
import dolfinx.fem.petsc
import ufl

def truss_stiffness(u, v, E, area):
    epsilon = lambda u: ufl.sym(ufl.grad(u))
    sigma = lambda u: ufl.tr(epsilon(u)) * ufl.Identity(len(u))
    
    return E * area * ufl.inner(sigma(u), epsilon(v)) * ufl.dx

def truss_mass(u, v, density, area_func):
    return density * area_func * ufl.inner(u, v) * ufl.dx

def plate_stiffness(u, v, E, nu, thickness):
    mu = E / (2 * (1 + nu))
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
        
    epsilon = lambda u: ufl.sym(ufl.grad(u))
    sigma = lambda u: lambda_ * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2.0 * mu * epsilon(u)
        
    return thickness * ufl.inner(sigma(u), epsilon(v)) * ufl.dx

def plate_mass(u, v, density, thickness):
    return density * thickness * ufl.inner(u, v) * ufl.dx

def mass_and_stiffness_matrix(mesh, props):
    V0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    V = dolfinx.fem.functionspace(mesh, ("CG", 1, (2,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    E_func = dolfinx.fem.Function(V0)
    E_func.x.array[:] = [material.young_modulus for material in props.materials.values()]
    
    density_func = dolfinx.fem.Function(V0)
    density_func.x.array[:] = [material.density for material in props.materials.values()]
    
    if mesh.topology.dim == 1:
        if not hasattr(props, 'diameters'):
            raise ValueError("add diameters_map in props")
        area_func = dolfinx.fem.Function(V0)
        area_func.x.array[:] = [np.pi * diameter**2 /4 for diameter in props.diameters.values()]
        
        stiffness_form = truss_stiffness(u, v, E_func, area_func)
        mass_form = truss_mass(u, v, density_func, area_func)
        
    else:
        nu_func = dolfinx.fem.Function(V0)
        nu_func.x.array[:] = [material.poisson_ratio for material in props.materials.values()]

        # Create thickness function, default to 1 if not specified
        thickness_func = dolfinx.fem.Function(V0)
        if hasattr(props, 'thicknesses'):
            thickness_func.x.array[:] = [thickness for thickness in props.thicknesses.values()]
        else:
            thickness_func.x.array[:] = np.ones(len(E_func.x.array))

        stiffness_form = plate_stiffness(u, v, E_func, nu_func, thickness_func)
        mass_form = plate_mass(u, v, density_func, thickness_func)

    mass_matrix = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(mass_form))
    mass_matrix.assemble()
    
    stiffness_matrix = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(stiffness_form))
    stiffness_matrix.assemble()
    
    return mass_matrix, stiffness_matrix

def global_matrixes(set_structure):
    """
    Create global mass and stiffness matrices from a SetStructure object.
    
    Args:
        set_structure: SetStructure object containing meshes and their properties
        
    Returns:
        tuple: (M_global, K_global) Global mass and stiffness matrices
    """
    mass_matrices = []
    stiffness_matrices = []
    sizes = []
    sparsity_patterns = []

    # Generate local matrices for each mesh and analyze sparsity patterns
    for mesh, prop in zip(set_structure.meshes, set_structure.props):
        mass, stiffness = mass_and_stiffness_matrix(mesh, prop)
        mass_matrices.append(mass)
        stiffness_matrices.append(stiffness)
        sizes.append(mass.getSize()[0])
        
        # Get sparsity pattern for this mesh's matrices
        pattern = {}
        for row in range(sizes[-1]):
            cols, _ = mass.getRow(row)
            pattern[row] = set(cols)  # Store column indices for each row
        sparsity_patterns.append(pattern)

    # Create global matrices with proper size
    total_size = set_structure.total_nodes * 2  # Multiply by 2 for x,y DOFs
    
    # Calculate preallocation arrays
    d_nnz = np.zeros(total_size, dtype=np.int32)  # Diagonal block nonzeros per row
    o_nnz = np.zeros(total_size, dtype=np.int32)  # Off-diagonal block nonzeros per row
    
    # Map local patterns to global patterns for preallocation
    offset = 0
    for mesh_idx, pattern in enumerate(sparsity_patterns):
        size = sizes[mesh_idx]
        for local_row in range(size):
            # Map local row to global row
            node_idx = local_row // 2
            mapped_node = set_structure.node_mapping[node_idx + offset]
            global_row = mapped_node * 2 + (local_row % 2)
            
            # Map local columns to global columns
            local_cols = pattern[local_row]
            for local_col in local_cols:
                col_node_idx = local_col // 2
                mapped_col_node = set_structure.node_mapping[col_node_idx + offset]
                global_col = mapped_col_node * 2 + (local_col % 2)
                
                # Update preallocation counts
                if abs(mapped_node - mapped_col_node) < size//2:
                    d_nnz[global_row] += 1
                else:
                    o_nnz[global_row] += 1
                    
        offset += size // 2

    # Create matrices with proper preallocation
    M_global = PETSc.Mat().createAIJ([total_size, total_size])
    K_global = PETSc.Mat().createAIJ([total_size, total_size])
    
    M_global.setPreallocationNNZ([d_nnz, o_nnz])
    K_global.setPreallocationNNZ([d_nnz, o_nnz])
    
    # Allow runtime allocation if our preallocation estimate is off
    M_global.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    K_global.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

    M_global.zeroEntries()
    K_global.zeroEntries()

    # Assembly loop
    offset = 0
    for mesh_idx, (mass, stiff) in enumerate(zip(mass_matrices, stiffness_matrices)):
        size = sizes[mesh_idx]
        
        for row in range(size):
            # Map local row to global row
            node_idx = row // 2
            mapped_node = set_structure.node_mapping[node_idx + offset]
            global_row = mapped_node * 2 + (row % 2)
            
            # Mass matrix
            cols, vals = mass.getRow(row)
            global_cols = []
            for col in cols:
                col_node_idx = col // 2
                mapped_col_node = set_structure.node_mapping[col_node_idx + offset]
                global_cols.append(mapped_col_node * 2 + (col % 2))
                
            M_global.setValues(global_row, global_cols, vals, addv=True)
            
            # Stiffness matrix
            cols, vals = stiff.getRow(row)
            global_cols = []
            for col in cols:
                col_node_idx = col // 2
                mapped_col_node = set_structure.node_mapping[col_node_idx + offset]
                global_cols.append(mapped_col_node * 2 + (col % 2))
                
            K_global.setValues(global_row, global_cols, vals, addv=True)
        
        offset += size // 2

    # Assemble final matrices
    M_global.assemblyBegin()
    M_global.assemblyEnd()
    K_global.assemblyBegin()
    K_global.assemblyEnd()

    return M_global, K_global