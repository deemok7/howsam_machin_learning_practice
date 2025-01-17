import numpy as np
from numpy import linalg as LA

np.set_printoptions(formatter={"all": lambda x: "{:.2g}".format(x)})


def calc_sigma(eVal, M):
    singular_values = np.sqrt(np.abs(eVal))
    singular_values = np.sort(singular_values)[::-1]

    m, n = M.shape
    sigma = np.zeros((m, n))

    np.fill_diagonal(sigma, singular_values[: min(m, n)])
    return sigma


def calc_svd(M, use_sorting, verbose):
    def m_print(s):
        if verbose:
            print(s)

    Mt = M.transpose()
    m_print("\n*** M^T ***")
    m_print(Mt)

    U = np.dot(M, Mt)
    U_eigenvalues, U_eigenvectors = LA.eig(U)
    m_print("\n*** U ***")
    m_print(f"U:\n{U}")
    m_print(f"U_eigenvalues:\n{U_eigenvalues}")
    m_print(f"U_eigenvectors:\n{U_eigenvectors}")

    V = np.dot(Mt, M)
    V_eigenvalues, V_eigenvectors = LA.eig(V)
    m_print("\n*** V ***")
    m_print(f"V:\n{V}")
    m_print(f"V_eigenvalues:\n{V_eigenvalues}")
    m_print(f"V_eigenvectors:\n{V_eigenvectors}")

    if use_sorting:
        m_print("\n*** Sorting ***")
        sorted_indices_U = np.argsort(-U_eigenvalues)
        sorted_indices_V = np.argsort(-V_eigenvalues)

        U_eigenvectors = U_eigenvectors[:, sorted_indices_U]
        V_eigenvectors = V_eigenvectors[:, sorted_indices_V]

        m_print(f"U_eigenvectors Sorted:\n{U_eigenvectors}")
        m_print(f"V_eigenvectors Sorted:\n{V_eigenvectors}")

    sigma = calc_sigma(U_eigenvalues, M)
    m_print("\n*** Sigma ***")
    m_print(sigma)

    # print("U_eigenvectors:",U_eigenvectors)
    # print("sigma", sigma)
    # print("V_eigenvectors", V_eigenvectors)
    return U_eigenvectors, sigma, V_eigenvectors


def test_svd(U, Sigma, V, M, verbose):
    if verbose:
        print("\n**************")
        print("*** Verify reconstruction of M ***")
        print("\nOriginal M:")
        print(M)
    V_T = V.transpose()
    # new_M = np.dot(U, np.dot(Sigma, V_T))
    new_M = np.matmul(U, np.matmul(Sigma, V_T))

    print("Reconstructed M:")
    print(new_M)


if __name__ == "__main__":
    verbose = False
    M_list = [
        np.array([[3, 2, 2], [2, 3, -2]]),
        np.array([[2, 4], [1, 3], [0, 0], [0, 0]]),
    ]

    for i, M in enumerate(M_list, start=1):
        print(f"\n======================")
        print(f"*** M {i} ***")
        print(M)

        m, n = M.shape
        if m == n:
            print("\nUsing Eigen")
            eigenvalues, eigenvectors = LA.eig(M)
            print(f"eigenvalues:\n{eigenvalues}")
            print(f"eigenvectors:\n{eigenvectors}")
        else:
            print("\nUsing SVD")
            use_sorting = True
            print(f"### use_sorting: {use_sorting}")
            U, sigma, V = calc_svd(M, use_sorting, verbose)
            test_svd(U, sigma, V, M, verbose)

            use_sorting = False
            print(f"### use_sorting: {use_sorting}")
            U, sigma, V = calc_svd(M, use_sorting, verbose)
            test_svd(U, sigma, V, M, verbose)

            # print(f"### Numpy SVD: ")
            # U, D, VT = np.linalg.svd(M)
            # print(U, D, VT)
    print()
