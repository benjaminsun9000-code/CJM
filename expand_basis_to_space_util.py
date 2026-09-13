import numpy as np
from rref_util import find_pivot_cols,rref
# input: nullspace of A basis and larger subspace basis
# output: M with new linearly independent vectors from N_A
def expand_basis_to_space(M, N_A):
  if M.size == 0 or M.shape[1] == 0:
    return N_A
  if N_A.size == 0 or N_A.shape[1] == 0:
        return M
  combined = np.hstack([M, N_A])
  R = rref(combined)
  R[np.abs(R) < 1e-8] = 0.0
  pivot_cols = find_pivot_cols(R)
  return combined[:,pivot_cols]
if __name__== "__main__":
    T = np.array([
    [2.0, 1.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 3.0]
    ])
    lamb = 2.0
    Q = np.array([
    [1.0, 0.0],
    [0.0, 0.0],
    [0.0, 1.0]
    ])
    M_hat = np.array([
        [1.0],
         [0.0]
    ])
    M = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0]
    ])
    N_A = np.array([
    [0.0],
    [1.0],
    [0.0]
    ])
    print(find_pivot_cols(rref(T)))
    print(expand_basis_to_space(M,N_A))
