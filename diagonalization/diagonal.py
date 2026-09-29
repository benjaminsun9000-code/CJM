# Item 1 -- How to diagonalize A = S LamMatrix S^-1, find the S, use left-inverse,outer product, biorthonormal

# Use Power Method to find a lambda
# check if A - lambda I has lower rank than (A-lambdaI)^2 : if it does, cant be eigendecomposed -- illustrate math why? how to find the algebraic multiplicity of lambda
# Find the corresponding eigenvectors for lambda by finding the basis for null space of (A-lambda I), which is S
# Compute Q^T which is the left inverse of S, which is (S^T S)^-1 S^T -- this is given partial S say n by k , find the biorthonormal matrix Q such that Q^T S = I k by k and S Q^T is a sum k outer product that A can be used to eliminate. This core idea is the same as SVD.
# The core idea of SVD is that one can view any linear transormation as sum of outer product matrix. Each of the outer product do projection value, scale by diagnolized value, and change direction
# Here for diagnalization of A using eigen vector, we follow similar patter to have A as S LamMatrix Q^T (think S ^-1 as Q^T). However, of each eigen value lambda, we have only partial S n by k matrix. Here k is the number of eigen vaectors (geo multiplicity) for this eigen value

# Item 2 If AB=BA and both diagonalizable, then they share same eigenvector matrix S. Find S
# Use Power Method to find a lambda
# check if A - lambda I has lower rank than (A-lambdaI)^2: if it does, cant be eigendecomposed
# Find the corresponding eigenvectors for lambda by finding the basis for null space of (A-lambdaI).
# If the eigenvalues are not distinct, then we define B hat as B but restricted onto the null space of (A-lambda I), or the eigenspace of lambda. 
# There is guranteed to be enough eigenvectors for us to fill, so we just find the eigenvectors of lambda.
# If the eigenvalues are distinct, then we have found enough eigenvectors when we find the basis for null space of (A-lambdaI).

# Prove Filippov's proof, when we get the M vectors, then they are linearly independent; use the chain vectors, extended vectors, null space completion vectors

import numpy as np
# input: matrix A
# output: dominant eigen value, eigen vector and A tranpose eigen vector
# Note 1: return error if can't find dominant eigen value
# Note 2: improve nilpotent case -- TODO
def power_dom_eigen(A, tol=1e-10, max_iter=1000000):
  A = np.asarray(A, dtype=float)
  N = A.shape[0]

  count_0 = 0
  LOOP_N = 5
  # to compensate for the p, possiblity of choosing initial vector perpendicular
  # to dominant eigen vector, we try five times
  for cnt in range(LOOP_N):
    q = np.random.rand(N)
    q = q / np.linalg.norm(q)
    for i in range(max_iter):
        q_new = A @ q
        norm_q = np.linalg.norm(q_new)
        if norm_q == 0:
          # The only eigen value is 0, which also means this matrix is [0] 
          count_0 += 1
          break
        q_new /= norm_q
        # print(f"i is {i}, q is {q}, q_new is {q_new},norm is {np.linalg.norm(q_new - q)}")
        if np.linalg.norm(q_new - q) < tol:
          lamb = (q_new.T @ A @ q_new) / (q_new.T @ q_new)
          return lamb, q_new
        q = q_new
  if (count_0 == LOOP_N):
    return 0
  # can't find one, return none
  return None
# A = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
# print(power_dom_eigen(A))

# input: rref reduced matrix
# output: indices of pivot column
def find_pivot_cols(R):
  rows, cols = R.shape
  pivot_cols = []
  for r in range(rows):
    for c in range(cols):
      if abs(R[r,c]) > 1e-10:
        if abs(R[r,c] - 1.0) < 1e-9:
          pivot_cols.append(c)
          break
  return pivot_cols

# input: matrix A
# output: rref form of A
def rref(A):
  A = np.asarray(A, dtype=float)
  (rows, cols)= A.shape
  r = 0
  for c in range(cols):
    pivot_row = None
    for i in range(r,rows):
      if abs(A[i][c]) > 1e-4:
        pivot_row = i
        break
    if pivot_row is None:
      continue
    A[r],A[pivot_row] = A[pivot_row].copy(),A[r].copy()
    # A[[r,pivot_row]] =A[[pivot_row, r]]
    pivot_val = A[r][c]
    A[r] = A[r] / pivot_val
    for i in range(rows):
      if i != r:
        factor = A[i][c]
        # A[i] = [a_i - factor * a_r for a_i, a_r in zip(A[i],A[r])]
        A[i] = A[i] - A[r] * factor
    # breakpoint()
    r += 1
    if r >= rows:
      break
  return A
# A = [
#    [ 2.0, -4.0,  6.0],
#    [-1.0,  2.0, -3.0],
#    [ 3.0, -6.0,  2.0],
#    [ 0.0,  0.0,  0.0],
#    [ 4.0, -8.0, 7.0]
# ]
# print(rref(A))
# input: matrix A
# output: null space basis for A
def basis_for_null_space(A):
  R = rref(A)
  R[np.abs(R) < 1e-8] = 0.0
  rows,cols = R.shape
  # print(R)
  pivot_cols=find_pivot_cols(R)
  free_cols = [c for c in range(cols) if c not in pivot_cols]
  if not free_cols:
    return np.empty((cols, 0))
  null_basis = []
  for free_col in free_cols:
    x = np.zeros(cols)
    x[free_col] = 1.0
    for r_idx,p_col in enumerate(pivot_cols):
      if r_idx < rows:
        x[p_col] = -R[r_idx,free_col]
    null_basis.append(x)
  return np.column_stack(null_basis)
#A = [
#   [ 1.0,  2.0, 0.0, -1.0],
#   [ 2.0,  4.0, 1.0,  1.0],
#    [ 3.0,  6.0, 1.0,  0.0]
#]

# input: a matrix S
# output: inverse of S
def invert_via_rref(S):
    n = S.shape[0]
    aug = np.hstack((S.astype(float), np.eye(n)))
    for i in range(n):
        max_row = i + np.argmax(np.abs(aug[i:, i]))
        aug[[i, max_row]] = aug[[max_row, i]]
        aug[i] = aug[i] / aug[i, i]
        for j in range(n):
            if i != j:
                aug[j] -= aug[j, i] * aug[i]
    return aug[:, n:]
# input: matrix A(diagonalizable)
# output: eigenvector matrix S, eigenvalue matrix Lambda, S^-1(inverse of eigenvector matrix S)
def diagonalized_matrix_parts(A,tol=1e-8):
    n = A.shape[0]
    A_current = A.astype(float)
    eigenvalues = []
    eigenvectors_list = []
    count = 0
    while count < n:
        lam, _ = power_dom_eigen(A_current)
        if lam is None:
          raise ValueError("Power Method failed to converge")
        M = A_current - lam * np.eye(n)
        rank_M = np.linalg.matrix_rank(M, tol=tol)
        rank_M2 = np.linalg.matrix_rank(M @ M, tol=tol)
        if rank_M > rank_M2:
          raise ValueError(f"Matrix is not diagonalizable {lam:.4f} has "
          f"generalized eigenvectors(Jordan block detected)")
        S1 = basis_for_null_space(M)
        k = S1.shape[1]
        for i in range(k):
            eigenvalues.append(lam)
            eigenvectors_list.append(S1[:, i])
        count += k
        Q1_T = np.linalg.inv(S1.T @ S1) @ S1.T
        A_current = A_current - lam * (S1 @ Q1_T)
    S = np.column_stack(eigenvectors_list)
    Lambda = np.diag(eigenvalues)
    S_inv = invert_via_rref(S)
    return S, Lambda, S_inv
A = np.array([[2, 1],
              [1, 2]])
print(diagonalized_matrix_parts(A))

B = np.array([
    [2., 0., 0., 0., 0., 3.],
    [0., 2., 0., 0., 3., -3.],
    [0., 0., 2., 0., 0., 3.],
    [0., 0., 0., 2., 3., -3.],
    [0., 0., 0., 0., 5., 0.],
    [0., 0., 0., 0., 0., 5.]
])
print(diagonalized_matrix_parts(B))

# not diagonalizable
C = np.array([
    [ 3,  1,  0],
    [ 0,  3,  1],
    [ 0,  0,  3]
])
try:
  S, Lambda, S_inv = diagonalized_matrix_parts(C)
  print(f"S is {S}")
  print(f"Lambda is {Lambda}")
  print(f"S_inv is {S_inv}")
  print(S @ Lambda @ S_inv)
  print(np.linalg.cond(S))
except (ValueError,TypeError):
  print("This matrix cannot be diagonalized")
