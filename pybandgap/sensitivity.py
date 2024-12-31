

def get_sensitivity_matrices(mesh, V, u):
    # Create material parameter field
    P0 = fem.FunctionSpace(mesh, ("DG", 0))
    alpha = fem.Function(P0)
    
    # Interpolation function from paper (eq. 2)
    E = (E2 - E1)*alpha + E1  # Young's modulus
    rho = (rho2 - rho1)*alpha + rho1  # Density

    # Bilinear forms
    u_test = ufl.TestFunction(V)
    
    # Stiffness form
    a = E*ufl.inner(ufl.grad(u), ufl.grad(u_test))*ufl.dx
    
    # Mass form  
    m = rho*ufl.inner(u, u_test)*ufl.dx

    # Get derivatives
    dK_dalpha = fem.assemble_matrix(ufl.derivative(a, alpha))
    dM_dalpha = fem.assemble_matrix(ufl.derivative(m, alpha))

    return dK_dalpha, dM_dalpha