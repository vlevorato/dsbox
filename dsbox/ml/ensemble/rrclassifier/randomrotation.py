import numpy as np

__credits__ = "https://github.com/tmadl/sklearn-random-rotation-ensembles"


def random_rot_mat_incl_flip(n):
    r = np.random.normal(size=(n, n))
    Q, R = np.linalg.qr(r)  # A=QR
    M = np.dot(Q, np.diag(np.sign(np.diag(R))))  # diag(R)>0
    return M.astype(np.float32)


def random_rotation_matrix(n):
    M = random_rot_mat_incl_flip(n)
    if np.linalg.det(M) < 0:
        M[:, 0] = -M[:, 0]  # det(M)=1
    return M
