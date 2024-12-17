import glob
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib import colors
from tensorflow.python.summary.summary_iterator import summary_iterator


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
    Psi_train,
    Psi_test,
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
            Psi_train[train_range[i]:train_range[i + 1]],
            '--C0'
        )
        plt.plot(
            D_test[test_range[i]:test_range[i + 1], D_idx],
            Psi_test[test_range[i]:test_range[i + 1]],
            '-.C1'
        )
    if func_of == 'D':
        plt.xlabel('$D_{%s}$' % D_comp)
    elif func_of == 'JD':
        plt.xlabel('$\overline{D}_{%s}$' % D_comp)
    plt.ylabel('$\Psi^D$ [GPa]')
    plt.legend(
        labels=['Ground truth', 'NAM (train/val)', 'NAM (test)'], loc='best'
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
    SD_train,
    SD_test,
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
            SD_train[train_range[i]:train_range[i + 1], SD_idx],
            '--C0'
        )
        plt.plot(
            D_test[test_range[i]:test_range[i + 1], D_idx],
            SD_test[test_range[i]:test_range[i + 1], SD_idx],
            '-.C1'
        )
    plt.xlabel('$D_{%s}$' % D_comp)
    plt.ylabel('$S_{%s}^D$ [GPa]' % SD_comp)
    plt.legend(
        labels=['Ground truth', 'NAM (train/val)', 'NAM (test)'], loc='best'
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


def plot_weight(weight, color, colorbar, plot_dir):
    colormap = mpl.colormaps[color].resampled(256)
    cmap = colors.ListedColormap(colormap(np.linspace(0, 1, num=256)))
    fig, ax = plt.subplots(figsize=(8, 6))
    psm = ax.imshow(weight, cmap=cmap, vmin=weight.min(), vmax=weight.max())
    ax.grid(which='major', axis='both', color='k', linestyle='-', linewidth=2)
    ax.set_xticks(np.arange(-0.5, weight.shape[1], 1))
    ax.set_yticks(np.arange(-0.5, weight.shape[0], 1))
    ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    if colorbar:
        fig.colorbar(psm, ax=ax)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/lin_weight.png')
    plt.close()


def plot_Dhatg(D_hat_train, D_hat_test, g_train, g_test, comp, plot_dir):
    idx = comp - 1
    plt.figure(figsize=(8, 6))
    plt.plot(D_hat_train[:, idx], g_train[:, idx], '.C0', markersize=5)
    plt.plot(D_hat_test[:, idx], g_test[:, idx], '.C1', markersize=2)
    plt.xlabel('$\widehat{D}_{%s}$' % comp)
    plt.ylabel('$g_{%s}$' % comp)
    plt.legend(labels=['NAM (train/val)', 'NAM (test)'], loc='best')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/Dhat{comp}_g{comp}.png')
    plt.close()


def plot_Dhatgp(D_hat_train, D_hat_test, gp_train, gp_test, comp, plot_dir):
    idx = comp - 1
    plt.figure(figsize=(8, 6))
    plt.plot(D_hat_train[:, idx], gp_train[:, idx], '.C0', markersize=5)
    plt.plot(D_hat_test[:, idx], gp_test[:, idx], '.C1', markersize=2)
    plt.xlabel('$\widehat{D}_{%s}$' % comp)
    plt.ylabel('$g^\prime_{%s}$' % comp)
    plt.legend(labels=['NAM (train/val)', 'NAM (test)'], loc='best')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/Dhat{comp}_gp{comp}.png')
    plt.close()


def plot_loss(time, plot_dir):
    logs = defaultdict(list)
    list_of_files = glob.glob(f'{time}/output/0/logs/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    for e in summary_iterator(latest_file):
        for v in e.summary.value:
            logs[v.tag].append(v.simple_value)
    plt.figure(figsize=(8, 6))
    plt.semilogy(
        range(1, len(logs['Logs/LossTrainEpoch']) + 1),
        logs['Logs/LossTrainEpoch'],
        '-C0',
        label='Train'
    )
    plt.semilogy(
        range(1, len(logs['Logs/LossValEpoch']) + 1),
        logs['Logs/LossValEpoch'],
        '--C1',
        label='Val'
    )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/loss.png')
    plt.close()


def plot_losses1(times, plot_dir):
    lines = []
    colors = ['C0', 'C1', 'C2']
    fig, ax = plt.subplots(figsize=(8, 6))
    legend = plt.figure(figsize=(10, 2))
    for k, time in enumerate(times):
        logs = defaultdict(list)
        list_of_files = glob.glob(f'{time}/output/0/logs/*')
        latest_file = max(list_of_files, key=os.path.getctime)
        for e in summary_iterator(latest_file):
            for v in e.summary.value:
                logs[v.tag].append(v.simple_value)
        l1, = ax.semilogy(
            range(1, len(logs['Logs/LossTrainEpoch']) + 1),
            logs['Logs/LossTrainEpoch'],
            f'-{colors[k]}',
            linewidth=2
        )
        l2, = ax.semilogy(
            range(1, len(logs['Logs/LossValEpoch']) + 1),
            logs['Logs/LossValEpoch'],
            f'--{colors[k]}',
            linewidth=1
        )
        lines.append(l1)
        lines.append(l2)        
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    legend.legend(
        handles=lines,
        labels=[
            'Train (fully conn.)',
            'Val (fully conn.)',
            'Train (NAM)',
            'Val (NAM)',
            'Train (HYDRA, $M = 6$)',
            'Val (HYDRA, $M = 6$)'
        ],
        loc='center',
        ncols=3
    )
    ax.grid()
    fig.tight_layout()
    legend.tight_layout()
    fig.savefig(f'{plot_dir}/losses1.png')
    legend.savefig(f'{plot_dir}/legend1.png')
    plt.close()


def plot_losses21(times, plot_dir):
    lines = []
    colors = ['C3', 'C4', 'C5']
    fig, ax = plt.subplots(figsize=(8, 6))
    legend = plt.figure(figsize=(10, 2))
    for k, time in enumerate(times):
        logs = defaultdict(list)
        list_of_files = glob.glob(f'{time}/output/0/logs/*')
        latest_file = max(list_of_files, key=os.path.getctime)
        for e in summary_iterator(latest_file):
            for v in e.summary.value:
                logs[v.tag].append(v.simple_value)
        l1, = ax.semilogy(
            range(1, len(logs['Logs/LossTrainEpoch']) + 1),
            logs['Logs/LossTrainEpoch'],
            f'-{colors[k]}',
            linewidth=2
        )
        l2, = ax.semilogy(
            range(1, len(logs['Logs/LossValEpoch']) + 1),
            logs['Logs/LossValEpoch'],
            f'--{colors[k]}',
            linewidth=1
        )
        lines.append(l1)
        lines.append(l2)        
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    legend.legend(
        handles=lines,
        labels=[
            'Train ($M = 3$)',
            'Val ($M = 3$)',
            'Train ($M = 2$)',
            'Val ($M = 2$)',
            'Train ($M = 1$)',
            'Val ($M = 1$)'
        ],
        loc='center',
        ncols=3
    )
    ax.grid()
    fig.tight_layout()
    legend.tight_layout()
    fig.savefig(f'{plot_dir}/losses2.png')
    legend.savefig(f'{plot_dir}/legend2.png')
    plt.close()


def plot_losses22(times, plot_dir):
    lines = []
    colors = ['C3', 'C4']
    fig, ax = plt.subplots(figsize=(8, 6))
    legend = plt.figure(figsize=(10, 2))
    for k, time in enumerate(times):
        logs = defaultdict(list)
        list_of_files = glob.glob(f'{time}/output/0/logs/*')
        latest_file = max(list_of_files, key=os.path.getctime)
        for e in summary_iterator(latest_file):
            for v in e.summary.value:
                logs[v.tag].append(v.simple_value)
        l1, = ax.semilogy(
            range(1, len(logs['Logs/LossTrainEpoch']) + 1),
            logs['Logs/LossTrainEpoch'],
            f'-{colors[k]}',
            linewidth=2
        )
        l2, = ax.semilogy(
            range(1, len(logs['Logs/LossValEpoch']) + 1),
            logs['Logs/LossValEpoch'],
            f'--{colors[k]}',
            linewidth=1
        )
        lines.append(l1)
        lines.append(l2)        
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    legend.legend(
        handles=lines,
        labels=[
            'Train ($M = 9$)',
            'Val ($M = 9$)',
            'Train ($M = 12$)',
            'Val ($M = 12$)'
        ],
        loc='center',
        ncols=2
    )
    ax.grid()
    fig.tight_layout()
    legend.tight_layout()
    fig.savefig(f'{plot_dir}/losses2.png')
    legend.savefig(f'{plot_dir}/legend2.png')
    plt.close()


def plot_losses23(times, plot_dir):
    lines = []
    colors = ['C3', 'C4', 'C5']
    fig, ax = plt.subplots(figsize=(8, 6))
    legend = plt.figure(figsize=(10, 2))
    for k, time in enumerate(times):
        logs = defaultdict(list)
        list_of_files = glob.glob(f'{time}/output/0/logs/*')
        latest_file = max(list_of_files, key=os.path.getctime)
        for e in summary_iterator(latest_file):
            for v in e.summary.value:
                logs[v.tag].append(v.simple_value)
        l1, = ax.semilogy(
            range(1, len(logs['Logs/LossTrainEpoch']) + 1),
            logs['Logs/LossTrainEpoch'],
            f'-{colors[k]}',
            linewidth=2
        )
        l2, = ax.semilogy(
            range(1, len(logs['Logs/LossValEpoch']) + 1),
            logs['Logs/LossValEpoch'],
            f'--{colors[k]}',
            linewidth=1
        )
        lines.append(l1)
        lines.append(l2)        
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    legend.legend(
        handles=lines,
        labels=[
            'Train ($M = 12$)',
            'Val ($M = 12$)',
            'Train ($M = 15$)',
            'Val ($M = 15$)',
            'Train ($M = 18$)',
            'Val ($M = 18$)'
        ],
        loc='center',
        ncols=3
    )
    ax.grid()
    fig.tight_layout()
    legend.tight_layout()
    fig.savefig(f'{plot_dir}/losses2.png')
    legend.savefig(f'{plot_dir}/legend2.png')
    plt.close()
