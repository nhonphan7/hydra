import numpy as np
import pandas as pd


def mat2voigt(mat):
    voigt = np.array([
        mat[0, 0], mat[1, 1], mat[2, 2], mat[1, 2], mat[0, 2], mat[0, 1]
    ])
    return voigt


def free_energy(D_voigt, cons2, cons3):
    Psi = 0.
    for i in range(6):
        for j in range(6):
            for k in range(6):
                Psi += (
                    0.5 * cons2[i, j] * D_voigt[i] * D_voigt[j]
                    + cons3[i, j, k] * D_voigt[i] * D_voigt[j] * D_voigt[k] / 6.
                )
    return Psi


def PK2_D(D_voigt, cons2, cons3):
    SD_voigt = np.zeros(6)
    for i in range(6):
        for j in range(6):
            for k in range(6):
                SD_voigt[i] += (
                    cons2[i, j] * D_voigt[j]
                    + 0.5 * cons3[i, j, k] * D_voigt[j] * D_voigt[k]
                )
    return SD_voigt


if __name__ == '__main__':
    strain = 0.05
    num = 1000
    C11 = 1050.     # GPa
    C12 = 127.      # GPa
    C44 = 550.      # GPa
    CD111 = 5570.   # GPa
    CD112 = -830.   # GPa
    CD123 = 670.    # GPa
    CD144 = 150.    # GPa
    CD166 = 780.    # GPa
    CD456 = 570.    # GPa

    cons2 = np.zeros((6, 6))
    cons2[0, 0] = cons2[1, 1] = cons2[2, 2] = C11   # 11, 22, 33
    cons2[0, 1] = cons2[1, 0] = C12                 # 12
    cons2[0, 2] = cons2[2, 0] = C12                 # 13
    cons2[1, 2] = cons2[2, 1] = C12                 # 23
    cons2[3, 3] = cons2[4, 4] = cons2[5, 5] = C44   # 44, 55, 66

    cons3 = np.zeros((6, 6, 6))
    cons3[0, 0, 0] = cons3[1, 1, 1] = cons3[2, 2, 2] = CD111    # 111, 222, 333
    cons3[0, 0, 1] = cons3[0, 1, 0] = cons3[1, 0, 0] = CD112    # 112
    cons3[0, 0, 2] = cons3[0, 2, 0] = cons3[2, 0, 0] = CD112    # 113
    cons3[0, 1, 1] = cons3[1, 1, 0] = cons3[1, 0, 1] = CD112    # 122
    cons3[0, 2, 2] = cons3[2, 2, 0] = cons3[2, 0, 2] = CD112    # 133
    cons3[1, 1, 2] = cons3[1, 2, 1] = cons3[2, 1, 1] = CD112    # 223
    cons3[1, 2, 2] = cons3[2, 2, 1] = cons3[2, 1, 2] = CD112    # 233
    cons3[0, 1, 2] = cons3[2, 1, 0] = CD123                     # 123
    cons3[1, 2, 0] = cons3[0, 2, 1] = CD123                     # 123
    cons3[2, 0, 1] = cons3[1, 0, 2] = CD123                     # 123
    cons3[0, 3, 3] = cons3[3, 3, 0] = cons3[3, 0, 3] = CD144    # 144
    cons3[1, 4, 4] = cons3[4, 4, 1] = cons3[4, 1, 4] = CD144    # 255
    cons3[2, 5, 5] = cons3[5, 5, 2] = cons3[5, 2, 5] = CD144    # 366
    cons3[0, 4, 4] = cons3[4, 4, 0] = cons3[4, 0, 4] = CD166    # 155
    cons3[0, 5, 5] = cons3[5, 5, 0] = cons3[5, 0, 5] = CD166    # 166
    cons3[1, 3, 3] = cons3[3, 3, 1] = cons3[3, 1, 3] = CD166    # 244
    cons3[1, 5, 5] = cons3[5, 5, 1] = cons3[5, 1, 5] = CD166    # 266
    cons3[2, 3, 3] = cons3[3, 3, 2] = cons3[3, 2, 3] = CD166    # 344
    cons3[2, 4, 4] = cons3[4, 4, 2] = cons3[4, 2, 4] = CD166    # 355
    cons3[3, 4, 5] = cons3[5, 4, 3] = CD456                     # 456
    cons3[4, 5, 3] = cons3[3, 5, 4] = CD456                     # 456
    cons3[5, 3, 4] = cons3[4, 3, 5] = CD456                     # 456

    case = []
    D11, D22, D33 = [], [], []
    D23, D13, D12 = [], [], []
    psi = []
    SD11, SD22, SD33 = [], [], []
    SD23, SD13, SD12 = [], [], []

    # case 10: uniaxial compression (11 direction)
    # and negative simple shear (12 direction)
    uniaxial = np.linspace(1. - strain, 1., num=num)
    shear = np.linspace(-strain, 0., num=num)
    for J, gamma in zip(uniaxial, shear):
        F = np.array([
            [J, gamma, 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        F_inv = np.linalg.inv(F)
        D = 0.5 * (np.eye(3) - F_inv @ F_inv.T)
        D_voigt = mat2voigt(D)
        Psi = free_energy(D_voigt, cons2, cons3)
        SD_voigt = PK2_D(D_voigt, cons2, cons3)

        case.append(10)
        D11.append(D_voigt[0]), D22.append(D_voigt[1]), D33.append(D_voigt[2])
        D23.append(D_voigt[3]), D13.append(D_voigt[4]), D12.append(D_voigt[5])
        psi.append(Psi)
        SD11.append(SD_voigt[0])
        SD22.append(SD_voigt[1])
        SD33.append(SD_voigt[2])
        SD23.append(SD_voigt[3])
        SD13.append(SD_voigt[4])
        SD12.append(SD_voigt[5])

    # case 11: uniaxial tension (11 direction)
    # and positive simple shear (12 direction)
    uniaxial = np.linspace(1., 1. + strain, num=num)
    shear = np.linspace(0., strain, num=num)
    for J, gamma in zip(uniaxial, shear):
        F = np.array([
            [J, gamma, 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ])
        F_inv = np.linalg.inv(F)
        D = 0.5 * (np.eye(3) - F_inv @ F_inv.T)
        D_voigt = mat2voigt(D)
        Psi = free_energy(D_voigt, cons2, cons3)
        SD_voigt = PK2_D(D_voigt, cons2, cons3)

        case.append(11)
        D11.append(D_voigt[0]), D22.append(D_voigt[1]), D33.append(D_voigt[2])
        D23.append(D_voigt[3]), D13.append(D_voigt[4]), D12.append(D_voigt[5])
        psi.append(Psi)
        SD11.append(SD_voigt[0])
        SD22.append(SD_voigt[1])
        SD33.append(SD_voigt[2])
        SD23.append(SD_voigt[3])
        SD13.append(SD_voigt[4])
        SD12.append(SD_voigt[5])
    
    data = {
        'case': case,
        'd11': D11,
        'd22': D22,
        'd33': D33,
        'd23': D23,
        'd13': D13,
        'd12': D12,
        'psi': psi,
        'sd11': SD11,
        'sd22': SD22,
        'sd33': SD33,
        'sd23': SD23,
        'sd13': SD13,
        'sd12': SD12
    }
    df = pd.DataFrame(data=data)
    df.to_csv('cubic_test.csv', index=False)
