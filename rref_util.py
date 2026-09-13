import numpy as np

# Configure NumPy to suppress scientific notation and format floats neatly
np.set_printoptions(precision=2, suppress=True, formatter={'float': '{:6.2f}'.format})

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
    r += 1
    if r >= rows:
      break
  return A

# simple test 
if __name__ == "__main__":
    print("simple test for RREF")
    A = [
    [ 2.0, -4.0,  6.0],
    [-1.0,  2.0, -3.0],
    [ 3.0, -6.0,  2.0],
    [ 0.0,  0.0,  0.0],
    [ 4.0, -8.0, 7.0]
    ]
    for row in rref(A):
        print(row)

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
# simple test
if __name__ == "__main__":
    print("test case for find_pivot_cols")
    matrix = np.array([
    [1, 2, -1,  3],
    [2, 4,  1, -3],
    [3, 6,  0,  0]
    ])
    print(find_pivot_cols(matrix))

# input: rref form of A
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
# simple test case
if __name__ == "__main__":
    print("test case for basis_for_null_space")
    A = [
    [ 1.0,  2.0, 0.0, -1.0],
   [ 2.0,  4.0, 1.0,  1.0],
    [ 3.0,  6.0, 1.0,  0.0]
    ]
    print(basis_for_null_space(A))

# input: matrices A b to solve Ax=b
# output: None if no solution, otherwise return x as one of the solutions
def solve_ax_b(A,b):
  M,N = A.shape
  augmented = np.hstack([A, b])
  R = rref(augmented)
  R[np.abs(R) < 1e-8] = 0.0
  pivot_cols = find_pivot_cols(R)
  if N in pivot_cols:
    return None
  x = np.zeros((N,1))
  for r_idx, pivot_col in enumerate(pivot_cols):
      if pivot_col < N:
          x[pivot_col,0] = R[r_idx, N]
  return x
# simple test case
if __name__ == "__main__":
    print("test case for solve_ax_b")
    C = np.array([
    [0,  2,  1],
    [1, -2, -3],
    [4, -4,  1]
    ])
    b = np.array([5, -4, 12]).reshape(3,1)
    print(solve_ax_b(C,b))
