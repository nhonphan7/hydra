import numpy as np
import pandas as pd


def free_energy(J, D_iso, K0, G0):
    Psi = K0 * (J - np.log(J) - 1.) + G0 * np.einsum('ij,ij', D_iso, D_iso)
    return Psi


if __name__ == '__main__':
    strain = 0.1
    num = 400
    K0 = 14.8   # GPa
    G0 = 7.37   # GPa

    case = []
    det_F = []
    D11, D22, D33 = [], [], []
    D23, D13, D12 = [], [], []
    D_iso11, D_iso22, D_iso33 = [], [], []
    D_iso23, D_iso13, D_iso12 = [], [], []
    psi = []

    # case 0: uniaxial compression (11 direction)
    uniaxial = np.linspace(1. - strain, 1., num=num)
    for J in uniaxial:
        F = np.array([
            [J, 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        F_inv = np.linalg.inv(F)
        D = 0.5 * (np.eye(3) - F_inv @ F_inv.T)
        D_iso = 0.5 * (np.eye(3) - J**(2 / 3) * F_inv @ F_inv.T)
        Psi = free_energy(J, D_iso, K0, G0)

        case.append(0)
        det_F.append(J)
        D11.append(D[0, 0]), D22.append(D[1, 1]), D33.append(D[2, 2])
        D23.append(D[1, 2]), D13.append(D[0, 2]), D12.append(D[0, 1])
        D_iso11.append(D_iso[0, 0])
        D_iso22.append(D_iso[1, 1])
        D_iso33.append(D_iso[2, 2])
        D_iso23.append(D_iso[1, 2])
        D_iso13.append(D_iso[0, 2])
        D_iso12.append(D_iso[0, 1])
        psi.append(Psi)

    # case 1: positive simple shear (12 direction)
    shear = np.linspace(0., strain, num=num)
    for gamma in shear:
        F = np.array([
            [1., gamma, 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        J = np.linalg.det(F)
        F_inv = np.linalg.inv(F)
        D = 0.5 * (np.eye(3) - F_inv @ F_inv.T)
        D_iso = 0.5 * (np.eye(3) - J**(2 / 3) * F_inv @ F_inv.T)
        Psi = free_energy(J, D_iso, K0, G0)

        case.append(1)
        det_F.append(J)
        D11.append(D[0, 0]), D22.append(D[1, 1]), D33.append(D[2, 2])
        D23.append(D[1, 2]), D13.append(D[0, 2]), D12.append(D[0, 1])
        D_iso11.append(D_iso[0, 0])
        D_iso22.append(D_iso[1, 1])
        D_iso33.append(D_iso[2, 2])
        D_iso23.append(D_iso[1, 2])
        D_iso13.append(D_iso[0, 2])
        D_iso12.append(D_iso[0, 1])
        psi.append(Psi)

    # case 2: negative simple shear (12 direction)
    shear = np.linspace(-strain, 0., num=num)
    for gamma in shear:
        F = np.array([
            [1., gamma, 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        J = np.linalg.det(F)
        F_inv = np.linalg.inv(F)
        D = 0.5 * (np.eye(3) - F_inv @ F_inv.T)
        D_iso = 0.5 * (np.eye(3) - J**(2 / 3) * F_inv @ F_inv.T)
        Psi = free_energy(J, D_iso, K0, G0)

        case.append(2)
        det_F.append(J)
        D11.append(D[0, 0]), D22.append(D[1, 1]), D33.append(D[2, 2])
        D23.append(D[1, 2]), D13.append(D[0, 2]), D12.append(D[0, 1])
        D_iso11.append(D_iso[0, 0])
        D_iso22.append(D_iso[1, 1])
        D_iso33.append(D_iso[2, 2])
        D_iso23.append(D_iso[1, 2])
        D_iso13.append(D_iso[0, 2])
        D_iso12.append(D_iso[0, 1])
        psi.append(Psi)

    # case 3: uniaxial tension (11 direction)
    uniaxial = np.linspace(1., 1. + strain, num=num)
    for J in uniaxial:
        F = np.array([
            [J, 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        F_inv = np.linalg.inv(F)
        D = 0.5 * (np.eye(3) - F_inv @ F_inv.T)
        D_iso = 0.5 * (np.eye(3) - J**(2 / 3) * F_inv @ F_inv.T)
        Psi = free_energy(J, D_iso, K0, G0)

        case.append(3)
        det_F.append(J)
        D11.append(D[0, 0]), D22.append(D[1, 1]), D33.append(D[2, 2])
        D23.append(D[1, 2]), D13.append(D[0, 2]), D12.append(D[0, 1])
        D_iso11.append(D_iso[0, 0])
        D_iso22.append(D_iso[1, 1])
        D_iso33.append(D_iso[2, 2])
        D_iso23.append(D_iso[1, 2])
        D_iso13.append(D_iso[0, 2])
        D_iso12.append(D_iso[0, 1])
        psi.append(Psi)

    # case 4: equibiaxial compression (11 and 22 directions)
    biaxial = np.linspace(1. - strain, 1., num=num)
    for J_sqrt in biaxial:
        F = np.array([
            [J_sqrt, 0., 0.],
            [0., J_sqrt, 0.],
            [0., 0., 1.]
        ])
        J = np.linalg.det(F)
        F_inv = np.linalg.inv(F)
        D = 0.5 * (np.eye(3) - F_inv @ F_inv.T)
        D_iso = 0.5 * (np.eye(3) - J**(2 / 3) * F_inv @ F_inv.T)
        Psi = free_energy(J, D_iso, K0, G0)

        case.append(4)
        det_F.append(J)
        D11.append(D[0, 0]), D22.append(D[1, 1]), D33.append(D[2, 2])
        D23.append(D[1, 2]), D13.append(D[0, 2]), D12.append(D[0, 1])
        D_iso11.append(D_iso[0, 0])
        D_iso22.append(D_iso[1, 1])
        D_iso33.append(D_iso[2, 2])
        D_iso23.append(D_iso[1, 2])
        D_iso13.append(D_iso[0, 2])
        D_iso12.append(D_iso[0, 1])
        psi.append(Psi)
    
    data = {
        'case': case,
        'j': det_F,
        'd11': D11,
        'd22': D22,
        'd33': D33,
        'd23': D23,
        'd13': D13,
        'd12': D12,
        'd_iso11': D_iso11,
        'd_iso22': D_iso22,
        'd_iso33': D_iso33,
        'd_iso23': D_iso23,
        'd_iso13': D_iso13,
        'd_iso12': D_iso12,
        'psi': psi
    }
    df = pd.DataFrame(data=data)
    df.to_csv('isotropic.csv', index=False)
