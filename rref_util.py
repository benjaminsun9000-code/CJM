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
