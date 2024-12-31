from dataclasses import dataclass
from pybandgap.set_structure import SetStructure
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py import PETSc

@dataclass
class FemUtils:
    structure: SetStructure
    
    def __post_init__(self):
        if len(self.structure.meshes) == 1:
            # Single mesh case
            mesh = self.structure.meshes[0]
            self.V0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
            self.V = dolfinx.fem.functionspace(mesh, ("CG", 1, (2,)))
            self.u = ufl.TrialFunction(self.V)
            self.v = ufl.TestFunction(self.V)
        else:
            # Multiple mesh case
            self.V0 = []
            self.V = []
            self.u = []
            self.v = []
            
            for mesh in self.structure.meshes:
                V0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
                V = dolfinx.fem.functionspace(mesh, ("CG", 1, (2,)))
                self.V0 += [V0]
                self.V += [V]
                self.u += [ufl.TrialFunction(V)]
                self.v += [ufl.TestFunction(V)]
    
    def assemble_global_matrix(self, forms):
        """
        Assemble global matrices from a form or list of forms.
        
        Parameters:
        forms: Single UFL form or list of UFL forms for each mesh
        
        Returns:
        PETSc.Mat: Assembled global matrix
        """
        # Handle single form case
        if not isinstance(forms, list):
            forms = [forms]
            V_list = [self.V] if isinstance(self.V, dolfinx.fem.FunctionSpace) else self.V
        else:
            V_list = self.V if isinstance(self.V, list) else [self.V]
            
        # Create global matrix with size based on total unique nodes
        total_dofs = self.structure.total_nodes * 2  # 2 DOFs per node (x,y components)
        A_global = PETSc.Mat().create()
        A_global.setSizes(((total_dofs, total_dofs)))
        A_global.setType('aij')
        A_global.setUp()
        
        # Assemble each mesh's contribution
        for mesh_idx, (form, V) in enumerate(zip(forms, V_list)):
            # Assemble local matrix
            A_local = dolfinx.fem.petsc.assemble_matrix(form)
            A_local.assemble()
            
            # Get DOF mapping for this mesh
            dof_map = V.dofmap
            num_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
            
            # Create mapping from local to global DOFs
            local_to_global = np.zeros(num_dofs, dtype=np.int32)
            for i in range(num_dofs // 2):  # Loop over nodes
                node_idx = i
                global_node = self.structure.global_indices[mesh_idx][node_idx]
                # Map x and y components
                local_to_global[2*i] = 2*global_node
                local_to_global[2*i + 1] = 2*global_node + 1
            
            # Transfer local matrix entries to global matrix
            for i in range(num_dofs):
                rows = A_local.getRow(i)[0]
                vals = A_local.getRow(i)[1]
                
                global_row = local_to_global[i]
                global_cols = local_to_global[rows]
                
                A_global.setValues(global_row, global_cols, vals, addv=PETSc.InsertMode.ADD)
        
        A_global.assemble()
        return A_global
    
    def assemble_global_vector(self, forms):
        """
        Assemble global vector from a form or list of forms.
        
        Parameters:
        forms: Single UFL form or list of UFL forms for each mesh
        
        Returns:
        PETSc.Vec: Assembled global vector
        """
        # Handle single form case
        if not isinstance(forms, list):
            forms = [forms]
            V_list = [self.V] if isinstance(self.V, dolfinx.fem.FunctionSpace) else self.V
        else:
            V_list = self.V if isinstance(self.V, list) else [self.V]
            
        total_dofs = self.structure.total_nodes * 2
        b_global = PETSc.Vec().create()
        b_global.setSizes(total_dofs)
        b_global.setUp()
        
        for mesh_idx, (form, V) in enumerate(zip(forms, V_list)):
            b_local = dolfinx.fem.petsc.assemble_vector(form)
            
            num_dofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
            local_to_global = np.zeros(num_dofs, dtype=np.int32)
            
            for i in range(num_dofs // 2):
                node_idx = i
                global_node = self.structure.global_indices[mesh_idx][node_idx]
                local_to_global[2*i] = 2*global_node
                local_to_global[2*i + 1] = 2*global_node + 1
            
            for i in range(num_dofs):
                global_idx = local_to_global[i]
                b_global.setValues(global_idx, b_local.getValues(i), addv=PETSc.InsertMode.ADD)
        
        b_global.assemble()
        return b_global