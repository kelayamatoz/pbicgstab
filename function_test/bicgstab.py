import numpy as np
import scipy.io as io
import copy


#  Algorithm 2.3 Biconjugate Gradient Stabilized (BICGSTAB)
# 1: Compute r0 =b−Ax0, choose r′0 such that r0·r′0 ̸=0
# 2: Set p0=r0
# 3: for j=0,1,···do
# 4: αj =(rj ·r′0)/((Apj)·r′0)
# 5: sj =rj −αjApj
# 6: ωj =((Asj)·sj)/((Asj)·(Asj))
# 7: xj+1 =xj +αjpj +ωjsj
# 8: rj+1 =sj −ωjAsj
# 9: if ||rj+1||<ε0 then
# 10: Break;
# 11: end if
# 12: βj =(αj/ωj)×(rj+1·r′0)/(rj·r′0)
# 13: pj+1 =rj+1+βj(pj −ωjApj)
# 14: end for
# 15: Setx=xj+1

def bicgstab(mat_file: str = '../gr_900_900_crg.mtx', max_iter: int = 1, epsilon: float = 1.e-14) -> None:
    A = io.mmread(mat_file)
    M, N = A.shape
    x = np.ones(N) * 0.1
    b = np.ones(N) * 0.2
    r0 = b - A @ x
    r0_prime = copy.copy(r0)
    p_j = copy.copy(r0)
    r_j = copy.copy(r0)
    for _ in range(max_iter):
        alpha_j = (r_j @ r0_prime) / (A @ p_j @ r0_prime)
        s_j = r_j - alpha_j * A @ p_j
        omega_j = ((A @ s_j) @ s_j) / ((A @ s_j) @ (A @ s_j))
        x_new = x + alpha_j * p_j + omega_j * s_j
        r_new = s_j - omega_j * A @ s_j
        if np.linalg.norm(r_new) < epsilon:
            break
        beta_j = (alpha_j / omega_j) * (r_new @ r0_prime) / (r_j @ r0_prime)
        p_new = r_new + beta_j * (p_j - omega_j * A @ p_j)
        x = x_new
        p = p_new

    print(x)


if __name__ == '__main__':
    bicgstab()
