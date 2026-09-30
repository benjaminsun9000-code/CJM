# Item 2 If AB=BA and both diagonalizable, then they share same eigenvector matrix S. Find S
# Use Power Method to find a lambda
# check if A - lambda I has lower rank than (A-lambdaI)^2: if it does, cant be eigendecomposed
# Find the corresponding eigenvectors for lambda by finding the basis for null space of (A-lambdaI).
# If the eigenvalues are not distinct, then we define B hat as B but restricted onto the null space of (A-lambda I), or the eigenspace of lambda. 
# There is guranteed to be enough eigenvectors for us to fill, so we just find the eigenvectors of lambda.
# If the eigenvalues are distinct, then we have found enough eigenvectors when we find the basis for null space of (A-lambdaI)

# input: rref reduced matrix
# output: indices of pivot column
import numpy as np
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
# output: make a copy of A and return rref form of A

def rref(A):
  A = np.array(A, dtype=float, copy=True)
  (rows, cols)= A.shape
  r = 0
  for c in range(cols):
    pivot_row = None
    for i in range(r,rows):
      if abs(A[i][c]) > 1e-9:
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
def diagonalized_matrix_parts(A, tol=1e-8):
    A = A.astype(float)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    if np.linalg.matrix_rank(eigenvectors, tol=tol) < A.shape[0]:
        raise ValueError("Matrix is not diagonalizable (defective matrix detected).")
    S = eigenvectors
    Lambda = np.diag(eigenvalues)
    S_inv = invert_via_rref(S)
    return S, Lambda, S_inv

# input: matrix B and basis V_lam
# output: B hat
def linear_transformation_onto_a_restricted_domain(B,V_lam):
  return np.linalg.inv(V_lam.T @ V_lam) @ V_lam.T @ B @ V_lam

def is_same_direction(v1, v2):
    # Calculate the dot product
    dot_product = np.dot(v1, v2)

    # Calculate the magnitudes (norms) of both vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Avoid division by zero for zero vectors
    if norm_v1 == 0 or norm_v2 == 0:
        return False

    # Calculate cosine similarity
    cosine_similarity = dot_product / (norm_v1 * norm_v2)

    # Check if cosine similarity is extremely close to 1
    return np.isclose(abs(cosine_similarity), 1.0)

# input: two matrices A and B
# output: common eigenvector matrix S
def find_shared_eigenvector_matrix(A, B, tol=1e-8):
  A = np.asarray(A, dtype=float)
  B = np.asarray(B, dtype=float)
  n = A.shape[0]
  S_columns = []

  eigen_vals, eigen_vecs = np.linalg.eig(A)
  eigen_set = set(eigen_vals)
  for lam in eigen_set:
    print(f"lam is {lam}")
    # breakpoint()
    M = A - lam * np.eye(n)
    R_M = rref(M)
    p_cols_M = find_pivot_cols(R_M)
    rank_M = len(p_cols_M)
    M2 = M @ M
    R_M2 = rref(M2)
    p_cols_M2 = find_pivot_cols(R_M2)
    rank_M2 = len(p_cols_M2)
    if rank_M > rank_M2:
      print(f"rank_M is {rank_M}, rank_M2 is {rank_M2} and M is {M} and M_2 is {M2}, A is {A}, lam is {lam}")
      raise ValueError(f"Matrix A is defective; eigenvalue {lam} has generalized eigenvectors.")
    V_lam = basis_for_null_space(M)
    k = V_lam.shape[1]
    # breakpoint()
    B_hat = linear_transformation_onto_a_restricted_domain(B,V_lam)
    V_hat, _, _ = diagonalized_matrix_parts(B_hat, tol=tol)
    V_shared = V_lam @ V_hat
    print(f" V_hat shape is {V_hat.shape}, V_lam shape is {V_lam.shape}")
    assert V_hat.shape[1] == k, "V_hat not same dim"
    # validate V_shared are indeed eigen vector of A and B
    for i in range(k):
      vec = V_shared[:,i]
      assert is_same_direction(vec, A @ vec), (f"A {A} vec { vec} not same direction")
      assert is_same_direction(vec, B @ vec), (f"B {B} vec { vec} not same direction")
    for i in range(k):
        S_columns.append(V_shared[:, i])
    S = np.column_stack(S_columns)
  return S

#A = np.array([
#    [-1.0,  2.0],
#    [-4.0,  5.0]
#])
#B = np.array([
#    [ 5.0, -3.0],
#    [ 6.0, -4.0]
#])

# S = find_shared_eigenvector_matrix(A,B)
#print(f"shared eigen vector matrix is {S} and checking each column is eigen vector of both A and B next")
#for i in range(S.shape[1]):
#  print(is_same_direction(S[:,i], A @ S[:,i]))
#  print(is_same_direction(S[:,i], B @ S[:,i]))
# bigger matrix test case
A = np.array([
    [2, 1, 0, 0, 1, 1],
    [1, 3, 1, 0, 0, 1],
    [0, 1, 2, 1, 0, 0],
    [0, 0, 1, 4, 1, 0],
    [1, 0, 0, 1, 2, 1],
    [1, 1, 0, 0, 1, 3]
], dtype=float)

B = np.array([
    [ 3,  4,  1,  1,  3,  5],
    [ 4,  6,  3,  1,  2,  5],
    [ 1,  3,  2,  4,  1,  1],
    [ 1,  1,  4, 10,  4,  1],
    [ 3,  2,  1,  4,  3,  4],
    [ 5,  5,  1,  1,  4,  6]
], dtype=float)
S = find_shared_eigenvector_matrix(A,B)
print(f"shared eigen vector matrix is {S} and checking each column is eigen vector of both A and B next")
for i in range(S.shape[1]):
  print(is_same_direction(S[:,i], A @ S[:,i]))
  print(is_same_direction(S[:,i], B @ S[:,i]))
