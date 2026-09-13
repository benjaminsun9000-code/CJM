import numpy as np
from power_dom_eigen_util import power_dom_eigen
from gram_schmidt_util import gram_schmidt
from expand_basis_to_space_util import expand_basis_to_space
from rref_util import rref,find_pivot_cols,basis_for_null_space,solve_ax_b
from T_hat_util import T_hat
from convert_j_to_blocks_util import convert_J_to_blocks,convert_blocks_to_jordan_form
# helper function, passing out easy to manipulate form equivalent to J
def cjm_(T):
  N = T.shape[0]

  # recursive base case
  assert(N >= 1)
  if N == 1:
    M = np.array([[1.0]])
    J = np.array([[T[0, 0]]])
    blocks = np.array([[0, 1, T[0, 0]]])
    return M, blocks
  
  # find one eigen value
  dominant_eigen = power_dom_eigen(T)
  if dominant_eigen is None:
    print("eigen value is not found")
    raise ValueError("eigenvalue can't be found")
  else:
    try:
      lamb = dominant_eigen[0]
    except (ValueError,IndexError,TypeError):
      lamb = dominant_eigen

  # restrict T to T-lamb * I column space as T hat  
  A = T - lamb * np.eye(N)
  Q,R = gram_schmidt(A)
  assert(Q.shape[1] != 0)
  T_h = T_hat(T,Q)
  #print(f"eigen value is {lamb}\n T_hat is {T_h}\n Q is {Q}")

  # recursive call cjm
  M_hat, blocks_hat = cjm_(T_h)
  
  # breakpoint()

  # convert M_hat from Q coordinates to {e} coordinates
  B = Q @ M_hat
  
  # find sub blocks corresponding to lamb and pull out one more vector in this
  # eigen vector chain
  chain = []
  new_block = []
  eigenvectors=[]
  current_idx = 0
  for start_idx,block_dim,b_lamb in blocks_hat:
    start_idx = int(start_idx)
    block_dim = int(block_dim)
    end_idx = start_idx + block_dim
    chain.extend([B[:,[i]] for i in range(start_idx,end_idx)])
    if abs(b_lamb - lamb) < 1e-3:
      eigenvectors.append(B[:,[start_idx]])
      b_last = B[:, [end_idx - 1]]
      x = solve_ax_b(A, b_last)
      if x is not None:
        chain.append(x)
        block_dim += 1
      else:
        raise ValueError("internal error: pull out should always work")
    new_block.append([current_idx, block_dim, b_lamb])
    current_idx += block_dim

  M_chains = np.hstack(chain)

  # find additional eigen vector in A (aka T - lamb*I) and expanding J 

  # it is possible the eigen value lamb's blocks are empty
  if len(eigenvectors) > 0:
    V = np.hstack(eigenvectors)
  else:
    V = np.empty((N,0))
  
  N_A = basis_for_null_space(A)
  null_A_basis = expand_basis_to_space(V,N_A)
  new_eigenvectors = null_A_basis[:,V.shape[1]:]
  M = np.hstack([M_chains,new_eigenvectors])
  for i in range(new_eigenvectors.shape[1]):
        new_block.append([current_idx, 1, lamb])
        current_idx += 1
  return M, np.array(new_block)

# input: square matrix T
# output: M,J in Jordan Form
def cjm(T):

  M, blocks = cjm_(T)
  J = convert_blocks_to_jordan_form(blocks)
  return M,J

# test case
print("test case for cjm")
T = np.array([[4.0, 1.0, -1.0], [0.0, 4.0, -3.0], [0.0, 0.0, 1.0]])
M,J = cjm(T)
print(f"M is {M}\n J is {J}")
print(f"test M J = T M as {np.allclose(M @ J,T @ M)}")
# Test case with eigenvalue as 0
T1 = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, -1.0], [0.0, 0.0, 3.0]])
M1,J1 = cjm(T1)
print(f"M is {M1}\n J is {J1}")
print(f"test M J = T M as {np.allclose(M1 @ J1, T1 @ M1)}")
