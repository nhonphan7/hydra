import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_DSD(df_np, case, case_idx, D_comp, D_idx, plot_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(df_np[case_idx, D_idx], df_np[case_idx, 49], label='$S_{11}^D$')
    plt.plot(df_np[case_idx, D_idx], df_np[case_idx, 50], label='$S_{12}^D$')
    plt.plot(df_np[case_idx, D_idx], df_np[case_idx, 51], label='$S_{13}^D$')
    plt.plot(df_np[case_idx, D_idx], df_np[case_idx, 52], label='$S_{21}^D$')
    plt.plot(df_np[case_idx, D_idx], df_np[case_idx, 53], label='$S_{22}^D$')
    plt.plot(df_np[case_idx, D_idx], df_np[case_idx, 54], label='$S_{23}^D$')
    plt.plot(df_np[case_idx, D_idx], df_np[case_idx, 55], label='$S_{31}^D$')
    plt.plot(df_np[case_idx, D_idx], df_np[case_idx, 56], label='$S_{32}^D$')
    plt.plot(df_np[case_idx, D_idx], df_np[case_idx, 57], label='$S_{33}^D$')
    plt.xlabel('$D_{%s}$' % D_comp)
    plt.ylabel('$S_{IJ}^D$ [GPa]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/case{case}_D{D_comp}_SD.png')
    plt.close()


if __name__ == '__main__':
    small_size = 20
    large_size = 24
    plt.rc('font', size=small_size)
    plt.rc('axes', labelsize=large_size)

    plot_dir = 'figures'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    df = pd.read_csv('../md_PF/all_Data_denoised_PF.csv')
    df = df.drop(df.columns[[0, 1]], axis=1)
    df_np = df.to_numpy()

    det_F = []
    sig11, sig12, sig13 = [], [], []
    sig21, sig22, sig23 = [], [], []
    sig31, sig32, sig33 = [], [], []
    F_inv11, F_inv12, F_inv13 = [], [], []
    F_inv21, F_inv22, F_inv23 = [], [], []
    F_inv31, F_inv32, F_inv33 = [], [], []
    D11, D12, D13 = [], [], []
    D21, D22, D23 = [], [], []
    D31, D32, D33 = [], [], []
    SD11, SD12, SD13 = [], [], []
    SD21, SD22, SD23 = [], [], []
    SD31, SD32, SD33 = [], [], []

    for i in range(df_np.shape[0]):
        F = np.array([
            [df_np[i, 3], df_np[i, 4], df_np[i, 5]],
            [df_np[i, 6], df_np[i, 7], df_np[i, 8]],
            [df_np[i, 9], df_np[i, 10], df_np[i, 11]]
        ])
        P = np.array([
            [df_np[i, 12], df_np[i, 13], df_np[i, 14]],
            [df_np[i, 15], df_np[i, 16], df_np[i, 17]],
            [df_np[i, 18], df_np[i, 19], df_np[i, 20]]
        ])

        J = np.linalg.det(F)
        sig = P @ F.T / J
        F_inv = np.linalg.inv(F)
        D = 0.5 * (np.eye(3) - F_inv @ F_inv.T)
        SD = J * F.T @ sig @ F

        det_F.append(J)
        sig11.append(sig[0, 0]), sig12.append(sig[0, 1]), sig13.append(sig[0, 2])
        sig21.append(sig[1, 0]), sig22.append(sig[1, 1]), sig23.append(sig[1, 2])
        sig31.append(sig[2, 0]), sig32.append(sig[2, 1]), sig33.append(sig[2, 2])
        F_inv11.append(F_inv[0, 0])
        F_inv12.append(F_inv[0, 1])
        F_inv13.append(F_inv[0, 2])
        F_inv21.append(F_inv[1, 0])
        F_inv22.append(F_inv[1, 1])
        F_inv23.append(F_inv[1, 2])
        F_inv31.append(F_inv[2, 0])
        F_inv32.append(F_inv[2, 1])
        F_inv33.append(F_inv[2, 2])       
        D11.append(D[0, 0]), D12.append(D[0, 1]), D13.append(D[0, 2])
        D21.append(D[1, 0]), D22.append(D[1, 1]), D23.append(D[1, 2])
        D31.append(D[2, 0]), D32.append(D[2, 1]), D33.append(D[2, 2])
        SD11.append(SD[0, 0]), SD12.append(SD[0, 1]), SD13.append(SD[0, 2])
        SD21.append(SD[1, 0]), SD22.append(SD[1, 1]), SD23.append(SD[1, 2])
        SD31.append(SD[2, 0]), SD32.append(SD[2, 1]), SD33.append(SD[2, 2])
    
    df['j'] = det_F
    df['sig11'], df['sig12'], df['sig13'] = sig11, sig12, sig13
    df['sig21'], df['sig22'], df['sig23'] = sig21, sig22, sig23
    df['sig31'], df['sig32'], df['sig33'] = sig31, sig32, sig33
    df['f_inv11'], df['f_inv12'], df['f_inv13'] = F_inv11, F_inv12, F_inv13
    df['f_inv21'], df['f_inv22'], df['f_inv23'] = F_inv21, F_inv22, F_inv23
    df['f_inv31'], df['f_inv32'], df['f_inv33'] = F_inv31, F_inv32, F_inv33
    df['d11'], df['d12'], df['d13'] = D11, D12, D13
    df['d21'], df['d22'], df['d23'] = D21, D22, D23
    df['d31'], df['d32'], df['d33'] = D31, D32, D33
    df['sd11'], df['sd12'], df['sd13'] = SD11, SD12, SD13
    df['sd21'], df['sd22'], df['sd23'] = SD21, SD22, SD23
    df['sd31'], df['sd32'], df['sd33'] = SD31, SD32, SD33
    df.to_csv('all_Data_denoised_eulerian.csv', index=False)
    df_np = df.to_numpy()
    
    for case in range(int(df['case'].max()) + 1):
        case_idx = df.index[df['case'] == case].to_list()

        plot_DSD(df_np, case, case_idx, 11, 40, plot_dir)
        plot_DSD(df_np, case, case_idx, 12, 41, plot_dir)
        plot_DSD(df_np, case, case_idx, 13, 42, plot_dir)
        plot_DSD(df_np, case, case_idx, 21, 43, plot_dir)
        plot_DSD(df_np, case, case_idx, 22, 44, plot_dir)
        plot_DSD(df_np, case, case_idx, 23, 45, plot_dir)
        plot_DSD(df_np, case, case_idx, 31, 46, plot_dir)
        plot_DSD(df_np, case, case_idx, 32, 47, plot_dir)
        plot_DSD(df_np, case, case_idx, 33, 48, plot_dir)
