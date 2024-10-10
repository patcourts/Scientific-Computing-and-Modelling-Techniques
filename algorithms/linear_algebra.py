def simpleBackwardSubstitution(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = b.size
    x = np.zeros_like(b)
    
    # First step
    x[n-1] = b[n-1]/U[n-1, n-1]
    
    # Main loop
    for i in range(n-2, -1, -1):
        h = 0 # helper variable
        for j in range (i+1, n):
            h += U[i, j]*x[j]
        x[i] = (b[i] - h)/U[i, i]

    return x

def simpleLU(A,nk):
    
    n = len(A)
    
    # Initialise L and U matrices
    U = A.copy()
    L = np.identity(n)
    
    # Main loop as in the lecture notes pseudocode
    for k in range(nk):   # for full LU factorisation, nk = n (nk added as a variable only for the animation)
        for j in range (k+1,n):
            L[j,k] = U[j,k]/U[k,k]
            U[j,k] = 0
            for i in range(k+1,n):
                U[j,i] = U[j,i] - L[j,k]*U[k,i]  
    return L, U