import sympy
import matplotlib.pyplot as plt


def write_eq(file, x, g_latex, g_eq, gp_eq, gpp_eq, loss):
    file.write(f'{x},{g_latex},{g_eq},{gp_eq},{gpp_eq},{loss}\n')


def construct_eq(func_of, voigt, lin_weight, g_eq, bias, factor=False):
    f = ''
    for k in range(lin_weight.shape[0]):
        z_hat = ''
        if func_of == 'JD':
            z_hat += f'+{lin_weight[k, 0]}*J'
            if factor:
                for i, comp in enumerate(voigt):
                    if comp in voigt[-3:]:
                        z_hat += f'+{lin_weight[k, i + 1]}*2*D{comp}'
                    else:
                        z_hat += f'+{lin_weight[k, i + 1]}*D{comp}'
            else:
                for i, comp in enumerate(voigt):
                    z_hat += f'+{lin_weight[k, i + 1]}*D{comp}'
        else:
            if factor:
                for i, comp in enumerate(voigt):
                    if comp in voigt[-3:]:
                        z_hat += f'+{lin_weight[k, i]}*2*{func_of}{comp}'
                    else:
                        z_hat += f'+{lin_weight[k, i]}*{func_of}{comp}'
            else:
                for i, comp in enumerate(voigt):
                    z_hat += f'+{lin_weight[k, i]}*{func_of}{comp}'
        g = g_eq[k].replace('exp', 'e')
        g = g.replace('x', '(' + z_hat + ')')
        g = g.replace('e', 'exp')
        f += f'+{g}'
    f += f'+{bias}'
    f = sympy.latex(sympy.sympify(f))
    return f


def comp2idx(comp):
    if comp == 11:
        idx = 0
    elif comp == 22:
        idx = 1
    elif comp == 33:
        idx = 2
    elif comp == 23:
        idx = 3
    elif comp == 13:
        idx = 4
    elif comp == 12:
        idx = 5
    return idx


def plot_DPsi(
    func_of,
    D,
    D_train,
    D_test,
    D_comp,
    Psi,
    Psi_nam_train,
    Psi_nam_test,
    Psi_sr_train,
    Psi_sr_test,
    train_range,
    test_range,
    plot_cases,
    plot_indices,
    plot_dir
):
    D_idx = comp2idx(D_comp)
    if func_of == 'JD':
        D_idx += 1
    case_idx = ''.join(map(str, plot_cases))
    plt.figure(figsize=(8, 6))
    for i in range(len(plot_indices)):
        plt.plot(D[plot_indices[i], D_idx], Psi[plot_indices[i]], '-k')
        plt.plot(
            D_train[train_range[i]:train_range[i + 1], D_idx],
            Psi_nam_train[train_range[i]:train_range[i + 1]],
            '--C0'
        )
        plt.plot(
            D_test[test_range[i]:test_range[i + 1], D_idx],
            Psi_nam_test[test_range[i]:test_range[i + 1]],
            '-.C1'
        )
        plt.plot(
            D_train[train_range[i]:train_range[i + 1], D_idx],
            Psi_sr_train[train_range[i]:train_range[i + 1]],
            ':C2'
        )
        plt.plot(
            D_test[test_range[i]:test_range[i + 1], D_idx],
            Psi_sr_test[test_range[i]:test_range[i + 1]],
            ':C2'
        )
    if func_of == 'D':
        plt.xlabel('$D_{%s}$' % D_comp)
    elif func_of == 'JD':
        plt.xlabel('$\overline{D}_{%s}$' % D_comp)
    plt.ylabel('$\Psi^D$ [GPa]')
    plt.legend(
        labels=['Ground truth', 'NAM (train/val)', 'NAM (test)', 'SR'],
        loc='best'
    )
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/case{case_idx}_D{D_comp}_Psi.png')
    plt.close()


def plot_DSDcomp(
    D,
    D_train,
    D_test,
    D_comp,
    SD,
    SD_nam_train,
    SD_nam_test,
    SD_sr_train,
    SD_sr_test,
    SD_comp,
    train_range,
    test_range,
    plot_cases,
    plot_indices,
    plot_dir
):
    D_idx = comp2idx(D_comp)
    SD_idx = comp2idx(SD_comp)
    case_idx = ''.join(map(str, plot_cases))
    plt.figure(figsize=(8, 6))
    for i in range(len(plot_indices)):
        plt.plot(D[plot_indices[i], D_idx], SD[plot_indices[i], SD_idx], '-k')
        plt.plot(
            D_train[train_range[i]:train_range[i + 1], D_idx],
            SD_nam_train[train_range[i]:train_range[i + 1], SD_idx],
            '--C0'
        )
        plt.plot(
            D_test[test_range[i]:test_range[i + 1], D_idx],
            SD_nam_test[test_range[i]:test_range[i + 1], SD_idx],
            '-.C1'
        )
        plt.plot(
            D_train[train_range[i]:train_range[i + 1], D_idx],
            SD_sr_train[train_range[i]:train_range[i + 1], SD_idx],
            ':C2'
        )
        plt.plot(
            D_test[test_range[i]:test_range[i + 1], D_idx],
            SD_sr_test[test_range[i]:test_range[i + 1], SD_idx],
            ':C2'
        )
    plt.xlabel('$D_{%s}$' % D_comp)
    plt.ylabel('$S_{%s}^D$ [GPa]' % SD_comp)
    plt.legend(
        labels=['Ground truth', 'NAM (train/val)', 'NAM (test)', 'SR'],
        loc='best'
    )
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/case{case_idx}_D{D_comp}_SD{SD_comp}.png')
    plt.close()


def plot_DSD(
    D,
    D_train,
    D_test,
    D_comp,
    SD,
    SD_train,
    SD_test,
    voigt,
    train_range,
    test_range,
    plot_cases,
    plot_indices,
    plot_dir
):
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    D_idx = comp2idx(D_comp)
    case_idx = ''.join(map(str, plot_cases))
    plt.figure(figsize=(8, 6))
    for k, SD_comp in enumerate(voigt):
        SD_idx = comp2idx(SD_comp)
        for i in range(len(plot_indices)):
            plt.plot(
                D[plot_indices[i], D_idx],
                SD[plot_indices[i], SD_idx],
                f'.{colors[k]}',
                markersize=10,
                markevery=1000
            )
            plt.plot(
                D_train[train_range[i]:train_range[i + 1], D_idx],
                SD_train[train_range[i]:train_range[i + 1], SD_idx],
                f'-{colors[k]}',
                label='$S_{%s}^D$' % SD_comp if i % 2 == 0 else '',
                linewidth=2
            )
            plt.plot(
                D_test[test_range[i]:test_range[i + 1], D_idx],
                SD_test[test_range[i]:test_range[i + 1], SD_idx],
                f'--{colors[k]}',
                linewidth=1
            )
    plt.xlabel('$D_{%s}$' % D_comp)
    plt.ylabel('$\\boldsymbol{S}^D$ [GPa]')
    plt.legend(loc='best', ncols=2)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/case{case_idx}_D{D_comp}_SD.png')
    plt.close()


def plot_Dhatg(
    D_hat_train,
    D_hat_test,
    D_hat_sr,
    g_nam_train,
    g_nam_test,
    g_sr,
    comp,
    plot_dir
):
    idx = comp - 1
    plt.figure(figsize=(8, 6))
    plt.plot(D_hat_train[:, idx], g_nam_train[:, idx], '.C0', markersize=5)
    plt.plot(D_hat_test[:, idx], g_nam_test[:, idx], '.C1', markersize=2)
    plt.plot(D_hat_sr[:, idx], g_sr[:, idx], '--C2', linewidth=2)
    plt.xlabel('$\widehat{D}_{%s}$' % comp)
    plt.ylabel('$g_{%s}$' % comp)
    plt.legend(labels=['NAM (train/val)', 'NAM (test)', 'SR'], loc='best')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/Dhat{comp}_g{comp}.png')
    plt.close()


def plot_Dhatgp(
    D_hat_train,
    D_hat_test,
    D_hat_sr,
    gp_nam_train,
    gp_nam_test,
    gp_sr,
    comp,
    plot_dir
):
    idx = comp - 1
    plt.figure(figsize=(8, 6))
    plt.plot(D_hat_train[:, idx], gp_nam_train[:, idx], '.C0', markersize=5)
    plt.plot(D_hat_test[:, idx], gp_nam_test[:, idx], '.C1', markersize=2)
    plt.plot(D_hat_sr[:, idx], gp_sr[:, idx], '--C2', linewidth=2)
    plt.xlabel('$\widehat{D}_{%s}$' % comp)
    plt.ylabel('$g^\prime_{%s}$' % comp)
    plt.legend(labels=['NAM (train/val)', 'NAM (test)', 'SR'], loc='best')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/Dhat{comp}_gp{comp}.png')
    plt.close()
