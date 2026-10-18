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
        if np.linalg.norm(q_new - q) < tol or np.linalg.norm(q_new+q) < tol:
          if np.linalg.norm(q_new + q) < tol and np.linalg.norm(q_new - q) >= tol:
            q_new = -q_new
          lamb = (q_new.T @ A @ q_new) / (q_new.T @ q_new)
          return lamb
        q = q_new
  if (count_0 == LOOP_N):
    return 0.0
  # can't find one, return none
  return None


# input: rref reduced matrix
# output: indices of pivot column
def find_pivot_cols(R):
  rows, cols = R.shape
  pivot_cols = []
  for r in range(rows):
    for c in range(cols):
      if abs(R[r, c]) > 1e-10:
        if abs(R[r, c] - 1.0) < 1e-9:
          pivot_cols.append(c)
          break
  return pivot_cols


# input: rref form of A
# output: null space basis for A
def basis_for_null_space(A):
  R = rref(A)
  R[np.abs(R) < 1e-8] = 0.0
  rows, cols = R.shape
  pivot_cols = find_pivot_cols(R)
  free_cols = [c for c in range(cols) if c not in pivot_cols]
  if not free_cols:
    return np.empty((cols, 0))
  null_basis = []
  for free_col in free_cols:
    x = np.zeros(cols)
    x[free_col] = 1.0
    for r_idx, p_col in enumerate(pivot_cols):
      if r_idx < rows:
        x[p_col] = -R[r_idx, free_col]
    null_basis.append(x)
  return np.column_stack(null_basis)


# input: matrix A
# output: rref form of A
def rref(A):
  A = np.array(A, dtype=float)
  (rows, cols) = A.shape
  r = 0
  for c in range(cols):
    pivot_row = None
    for i in range(r, rows):
      if abs(A[i][c]) > 1e-8:
        pivot_row = i
        break
    if pivot_row is None:
      continue
    A[r], A[pivot_row] = A[pivot_row].copy(), A[r].copy()
    pivot_val = A[r][c]
    A[r] = A[r] / pivot_val
    for i in range(rows):
      if i != r:
        factor = A[i][c]
        A[i] = A[i] - A[r] * factor
    r += 1
    if r >= rows:
      break
  A[np.abs(A) < 1e-8] = 0.0
  return A


# input: E and LN
# output: basis S and Q such that S^T Q = I
def dual_gram_schmidt(E, LN):
  N, k = E.shape
  Q = np.zeros((N, k))
  S = np.zeros((N, k))
  S[:, 0] = E[:, 0]
  d1 = LN[:, 0] @ S[:, 0]
  Q[:, 0] = LN[:, 0] / d1
  for j in range(1, k):
    s = E[:, j].copy()
    for i in range(j):
      s -= (Q[:, i] @ E[:, j]) * S[:, i]
    S[:, j] = s
    q = LN[:, j].copy()
    for i in range(j):
      q -= (S[:, i] @ LN[:, j]) * Q[:, i]
    d = q @ S[:, j]
    Q[:, j] = q / d
  return S, Q
E = np.array([
    [1.0, 2.0],
    [3.0, 1.0],
    [0.0, 1.0]
])

Ln = np.array([
    [1.0, 1.0],
    [2.0, 0.0],
    [-1.0, 1.0]
])
#print(dual_gram_schmidt(E,Ln))

# input: matrix A
# output: diagonalization of A (S Lambda S^-1)
def diagonalization(A, tol=1e-10, max_iter=1000000):
  A = A.astype(float)
  N = A.shape[0]
  Q_blocks = []
  S_blocks = []
  lambda_diag = []
  distinct_lambdas = []
  dims_found = 0
  E_i = basis_for_null_space(A)
  LN_i = basis_for_null_space(A.T)
  if E_i.shape[1] > 0:
    lam = 0.0
    distinct_lambdas.append(lam)
    S_i, Q_i = dual_gram_schmidt(E_i, LN_i)
    S_blocks.append(S_i)
    Q_blocks.append(Q_i)
    mult = E_i.shape[1]
    lambda_diag.extend([lam] * mult)
    dims_found += mult
  while dims_found < N:
    print(f"current A is {A}")
    lam = power_dom_eigen(A, tol=tol, max_iter=max_iter)
    if lam is not None and abs(lam - round(lam)) < 1e-3:
      lam = round(lam)
    print(f"found eigen value {lam}")
    if lam is None or abs(lam) < 1e-3:
      break
    distinct_lambdas.append(lam)
    #breakpoint()
    E_i = basis_for_null_space(A - lam * np.eye(N))
    LN_i = basis_for_null_space(A.T - lam * np.eye(N))
    S_i, Q_i = dual_gram_schmidt(E_i, LN_i)
    S_blocks.append(S_i)
    Q_blocks.append(Q_i)
    mult = E_i.shape[1]
    lambda_diag.extend([lam] * mult)
    dims_found += mult
    A -= lam * (S_i @ Q_i.T)
  if dims_found < N:
    raise RuntimeError(f"Matrix is defective: found only {dims_found} linearly independent eigenvectors out of {N}.")
  Q = np.column_stack(Q_blocks)
  S = np.column_stack(S_blocks)
  Lambda = np.diag(lambda_diag)
  return S, Lambda, Q.T

# Test case
A = np.array([
    [2., 0., 0., 0., 0., 3.],
    [0., 2., 0., 0., 3., -3.],
    [0., 0., 2., 0., 0., 3.],
    [0., 0., 0., 2., 3., -3.],
    [0., 0., 0., 0., 5., 0.],
    [0., 0., 0., 0., 0., 5.]
])
S, Lambda, S_inv = diagonalization(A)
#print(S)
#print(Lambda)
#print(S_inv)
print(np.allclose(A, S @ Lambda @ S_inv))
# Test case(lambda is 0)
A = np.array([
    [4.0, 1.0, 0.0],
    [0.0, 2.0, 0.0],
    [0.0, 0.0, 0.0]
])
S, Lambda, S_inv = diagonalization(A)
#print(S)
#print(Lambda)
#print(S_inv)
print(np.allclose(A, S @ Lambda @ S_inv))
