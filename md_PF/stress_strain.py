import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_FP(df_np, case, case_idx, F_comp, F_idx, plot_dir):
    plt.figure(figsize=(8, 6))
    plt.plot(df_np[case_idx, F_idx], df_np[case_idx, 12], label='$P_{11}$')
    plt.plot(df_np[case_idx, F_idx], df_np[case_idx, 13], label='$P_{12}$')
    plt.plot(df_np[case_idx, F_idx], df_np[case_idx, 14], label='$P_{13}$')
    plt.plot(df_np[case_idx, F_idx], df_np[case_idx, 15], label='$P_{21}$')
    plt.plot(df_np[case_idx, F_idx], df_np[case_idx, 16], label='$P_{22}$')
    plt.plot(df_np[case_idx, F_idx], df_np[case_idx, 17], label='$P_{23}$')
    plt.plot(df_np[case_idx, F_idx], df_np[case_idx, 18], label='$P_{31}$')
    plt.plot(df_np[case_idx, F_idx], df_np[case_idx, 19], label='$P_{32}$')
    plt.plot(df_np[case_idx, F_idx], df_np[case_idx, 20], label='$P_{33}$')
    plt.xlabel('$F_{%s}$' % F_comp)
    plt.ylabel('$P_{iJ}$ [GPa]')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/case{case}_F{F_comp}_P.png')
    plt.close()


if __name__ == '__main__':
    small_size = 20
    large_size = 24
    plt.rc('font', size=small_size)
    plt.rc('axes', labelsize=large_size)

    plot_dir = 'figures'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    df = pd.read_csv('all_Data_denoised_PF.csv')
    df = df.drop(df.columns[[0, 1]], axis=1)
    df_np = df.to_numpy()
    
    for case in range(int(df['case'].max()) + 1):
        case_idx = df.index[df['case'] == case].to_list()
        
        plot_FP(df_np, case, case_idx, 11, 3, plot_dir)
        plot_FP(df_np, case, case_idx, 12, 4, plot_dir)
        plot_FP(df_np, case, case_idx, 13, 5, plot_dir)
        plot_FP(df_np, case, case_idx, 21, 6, plot_dir)
        plot_FP(df_np, case, case_idx, 22, 7, plot_dir)
        plot_FP(df_np, case, case_idx, 23, 8, plot_dir)
        plot_FP(df_np, case, case_idx, 31, 9, plot_dir)
        plot_FP(df_np, case, case_idx, 32, 10, plot_dir)
        plot_FP(df_np, case, case_idx, 33, 11, plot_dir)
