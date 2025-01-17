import numpy as np
from numpy import linalg as LA

np.set_printoptions(formatter={"all": lambda x: "{:.2g}".format(x)})

def calc_sigma(eValU, eValV, M):
    singular_values = np.sqrt(np.abs(eValV))  # No matter if using eValU or eValV
    singular_values = np.sort(singular_values)[::-1]

    m, n = M.shape
    sigma = np.zeros((m, n))

    np.fill_diagonal(sigma, singular_values[: min(n, m)])

    return sigma

def calc_svd(M,use_sorting,verbose):
    def m_print(s):
        if verbose:
            print(s)
    # use SVD
    # U = M M^T
    # V = M^T M
    # M <=> Ub Sigma Vb^T

    Mt = M.transpose()
    
    m_print("\n*** M^T ***")
    m_print(Mt)

    U = np.matmul(M, Mt)
    U_eigenvalues, U_eigenvectors = LA.eig(U)
    m_print("\n*** U ***")
    m_print(f"U:\n{U}")
    m_print(f"U_eigenvalues:\n{U_eigenvalues}")
    m_print(f"U_eigenvectors:\n{U_eigenvectors}")

    V = np.matmul(Mt, M)
    V_eigenvalues, V_eigenvectors = LA.eig(V)
    m_print("\n*** V ***")
    m_print(f"V:\n{V}")
    m_print(f"V_eigenvalues:\n{V_eigenvalues}")
    m_print(f"V_eigenvectors:\n{V_eigenvectors}")

    if use_sorting:
        m_print("\n*** Sorting ***")
        # Sort eigenvectors of U and V according to eigenvalues
        sorted_indices_U = np.argsort(-U_eigenvalues)  # Descending order
        sorted_indices_V = np.argsort(-V_eigenvalues)  # Descending order

        U_eigenvectors = U_eigenvectors[:, sorted_indices_U]  # U eigenvectors
        V_eigenvectors = V_eigenvectors[:, sorted_indices_V]  # V eigenvectors

        m_print(f"U_eigenvectors Sorted:\n{U_eigenvectors}")
        m_print(f"V_eigenvectors Sorted:\n{V_eigenvectors}")

    sigma = calc_sigma(U_eigenvalues, V_eigenvalues, M)
    m_print("\n*** Sigma ***")
    m_print(f"{sigma}")

    return U_eigenvectors, sigma, V_eigenvectors

def test_svd(U_eigenvectors, Sigma, V_eigenvectors,verbose):
    if verbose:
        print("\n**************")
        print("\n\n*** Verify reconstruction of M ***")
    V_eigenvectors_T = V_eigenvectors.transpose()
    if verbose:
        print("\n*** V_eigenvectors^T ***")
        print(V_eigenvectors_T)

    new_M =-1* np.matmul(U_eigenvectors, np.matmul(Sigma, V_eigenvectors_T))
    if verbose:
        print(f"\norg M:\n{M}")
    print(f"reconstruction M:\n{new_M}")

if __name__ == "__main__":
    verbose=False
    M_list=[]
    
    # M_list.append( np.array([[1, 0.5], [2, 0.8]]))
    M_list.append( np.array([[3, 2, 2], [2, 3, -2]]))
    M_list.append( np.array([[2, 4], [1, 3], [0, 0], [0, 0]]))

    for i, M in enumerate(M_list,start=1):
        print(f"\n======================")
        print(f"*** M {i} ***")
        print(M)

        m, n = M.shape
        if m == n:
            print("\nUsing Eigen")
            eigenvalues, eigenvectors = LA.eig(M)
            print(f"eigenvalues:{eigenvalues}")
            print(f"eigenvectors:{eigenvectors}")
        else:
            print("\nUsing SVD")
            use_sorting=True
            print(f"### use_sorting: {use_sorting}")
            U_eigenvectors, sigma, V_eigenvectors = calc_svd(M,use_sorting,verbose)
            test_svd(U_eigenvectors, sigma, V_eigenvectors,verbose)
            
            use_sorting=False
            print(f"### use_sorting: {use_sorting}")
            U_eigenvectors, sigma, V_eigenvectors = calc_svd(M,use_sorting,verbose)
            test_svd(U_eigenvectors, sigma, V_eigenvectors,verbose)

    print()