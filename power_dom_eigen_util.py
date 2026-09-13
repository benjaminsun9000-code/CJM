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
        if np.linalg.norm(q_new - q) < tol:
          lamb = (q_new.T @ A @ q_new) / (q_new.T @ q_new)
          return lamb, q_new
        q = q_new
  if (count_0 == LOOP_N):
    return 0
  # can't find one, return none
  return None
# simple test
if __name__ == "__main__":
    print("test case for power_dom_eigen")
    A = np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    print(power_dom_eigen(A))
