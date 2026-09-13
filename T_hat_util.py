import numpy as np
# input: T is linear transform from Rn --> Rm, P's columns are linear independent vectors in Rn
# output: T hat, which is T but restricted on the column space of P
# T hat as a linear transforms' matrix, the ith column is the vector of T hat @ pi{p} under {p} basis
# T hat @ pi{p} is the same abstract vector of T @ pi{e}. That we have P (T hat @ pi{p}) = T @ pi{e}}
# Thus, the ith column of T hat (under {p} basis) is P+ P T pi{e}. Aka T hat = P+ P T.
# P+ is psuedo inverse of P. Since P has full column rank, P+ is (P.T @ P)^-1 @ P.T.
# Thus T hat is (P.T @ P)^-1 @ P.T @ T @ P
def T_hat(T,Q):
  return np.linalg.inv(Q.T @ Q) @ Q.T @ T @ Q
# simple test case
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
