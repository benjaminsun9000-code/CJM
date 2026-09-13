import numpy as np
# input: J
# output: an array with each row giving jordan block info[start_index, dimension, eigenvalue]
def convert_J_to_blocks(J):
  N = J.shape[0]
  # if dimension of J is (0,0) return None
  if N == 0:
    return None
  blocks = []
  start_idx = 0
  dim = 1
  for i in range(N):
    if i < N - 1 and abs(J[i,i+1] - 1.0) < 1e-10 and abs(J[i,i] - J[i+1,i+1]) < 1e-10:
      dim += 1
    else:
      eigenvalue = J[i,i]
      blocks.append([start_idx,dim,eigenvalue])
      start_idx = i + 1
      dim = 1
  return np.array(blocks)
# simple test case
if __name__ == "__main__":
    print("test case for convert_J_to_blocks")
    J = np.array(
    [
        [3.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 5.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 5.0],
    ]
    )
    print(convert_J_to_blocks(J))

# input: an array with each row giving jordan block info[start_index, dimension, eigenvalue]
# output: J

def convert_blocks_to_jordan_form(blocks):
  N = int(sum(block[1] for block in blocks))
  J = np.zeros((N,N))
  for start_idx,dim,eigenvalue in blocks:
    start_idx = int(round(start_idx))
    dim = int(round(dim))
    end_idx = start_idx + dim
    np.fill_diagonal(J[start_idx:end_idx,start_idx:end_idx],eigenvalue)
    if dim > 1:
      np.fill_diagonal(J[start_idx:end_idx - 1, start_idx+1:end_idx],1.0)
  return J
# simple test case
if __name__ == "__main__":
    print("test case for convert_blocks_to_jordan_form")
    blocks = np.array([[0, 2, 3.0], [2, 1, 1.0], [3, 2, 5.0]])
    print(convert_blocks_to_jordan_form(blocks))
