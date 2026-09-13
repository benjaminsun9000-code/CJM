import numpy as np

# input: matrix A
# calculate A column space's orthornomal basis
# output: orthornomal basis in Q matrix, and R is the Q decomposition
# the ith column of R is the coordinates of ith column of A in Q
# Configure NumPy to suppress scientific notation and format floats neatly
np.set_printoptions(precision=2, suppress=True, formatter={'float': '{:6.2f}'.format})
def gram_schmidt(A):
  M, N = A.shape
  Q = np.zeros((M, N))
  R = np.zeros((N, N))
  rank = 0
  for j in range(N):
      v = A[:, j].copy()
      for i in range(rank):
          R[i, j] = Q[:, i] @ A[:,j]
          v = v - R[i, j] * Q[:, i]
      norm_v = np.linalg.norm(v)
      if norm_v > 1e-4:
          Q[:, rank] = v / norm_v
          R[rank,j] = norm_v
          rank += 1
  return Q[:,:rank], R[:rank,:]
# test case for Gram_schmidt
if __name__ == "__main__":
    print("test case for gram schmidt")
    V = np.array([[1, 1, 1],
              [-1, 0, 2],
              [1, 2, 1]]).T
    print(gram_schmidt(V))

# input: T is linear transform from Rn --> Rm, P's columns are linear independent vectors in Rn
# output: T hat, which is T but restricted on the column space of P
# T hat as a linear transforms' matrix, the ith column is the vector of T hat @ pi{p} under {p} basis
# T hat @ pi{p} is the same abstract vector of T @ pi{e}. That we have P (T hat @ pi{p}) = T @ pi{e}}
# Thus, the ith column of T hat (under {p} basis) is P+ P T pi{e}. Aka T hat = P+ P T.
# P+ is psuedo inverse of P. Since P has full column rank, P+ is (P.T @ P)^-1 @ P.T.
# Thus T hat is (P.T @ P)^-1 @ P.T @ T @ P
def T_hat(T,Q):
  return np.linalg.inv(Q.T @ Q) @ Q.T @ T @ Q
# test case
if __name__ == "__main__":
    print("test case for T_hat")
    T = np.array([
    [ 4.0, -2.0,  1.0, 3.0],
    [ 0.0,  5.0, -1.0, 2.0],
    [-3.0,  1.0,  0.0, 6.0],
    [ 2.0,  4.0, -5.0, 1.0]
])
    Q,R = gram_schmidt(T)
    print(T_hat(T,Q))
    v_underpbasis = np.array([
    [ 1.0],
    [ 2.0],
    [ 0.0],
    [ 0.0]
])
    v_underebasis = Q @ v_underpbasis
    print(np.allclose(Q @ T_hat(T,Q) @ v_underpbasis, T @ v_underebasis))
