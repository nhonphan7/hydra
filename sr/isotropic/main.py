import datetime
import joblib
import os
import pysr
import sympy
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(1, '../')
from utils import *


if __name__ == '__main__':
    # Synthetic isotropic data
    # case 0: uniaxial compression (11 direction)
    # case 1: positive simple shear (12 direction)
    # case 2: negative simple shear (12 direction)
    # case 3: uniaxial tension (11 direction)
    # case 4: equibiaxial compression (11 and 22 directions)

    data_dir = '../../nam/isotropic/isotropic_D3.csv'
    # data_dir = '../../nam/isotropic/isotropic_JD2.csv'
    plot_cases = [0, 3]
    strain_comp = 11

    test_split = 0.15
    timeout_in_seconds = 60 * 10
    fit_step = 1
    func_of = 'D'
    # func_of = 'JD'
    scale = False

    small_size = 20
    large_size = 24
    plt.rc('font', size=small_size)
    plt.rc('axes', labelsize=large_size)

    time = datetime.datetime.now().isoformat()
    plot_dir = f'{time}/figures'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if scale and not os.path.exists('scales'):
        os.makedirs('scales')

    voigt = [11, 22, 33, 23, 13, 12]
    df = pd.read_csv(data_dir)
    if func_of == 'D':
        X = df[['d11', 'd22', 'd33', 'd23', 'd13', 'd12']].to_numpy()
        y = df['psi'].to_numpy()
    elif func_of == 'JD':
        X = df[[
            'j', 'd_iso11', 'd_iso22', 'd_iso33', 'd_iso23', 'd_iso13', 'd_iso12'
        ]].to_numpy()
        y = df['psi'].to_numpy()
    energy_nam = df['psi_nam'].to_numpy()
    g = df[['g1', 'g2', 'g3']].to_numpy()
    lin_weight = np.load('../../nam/isotropic/lin_weight_D3.npy')
    # g = df[['g1', 'g2']].to_numpy()
    # lin_weight = np.load('../../nam/isotropic/lin_weight_JD2.npy')
    if scale:
        X_scaler = MinMaxScaler()
        X_scale = X_scaler.fit_transform(X)
        joblib.dump(X_scaler, 'scales/X_scaler.pkl')
        y_scaler = MinMaxScaler()
        y_scale = y_scaler.fit_transform(y)
        joblib.dump(y_scaler, 'scales/y_scaler.pkl')
        nam_inputs = X_scale @ lin_weight.T
    else:
        nam_inputs = X @ lin_weight.T
    
    train_indices, test_indices = [], []
    plot_indices, train_plot_indices, test_plot_indices = [], [], []
    train_plot_len, test_plot_len = [], []
    for case in range(int(df['case'].max()) + 1):
    # for case in plot_cases:
        case_idx = df.index[df['case'] == case].to_list()
        if case % 2 == 0:
            train_case_idx = case_idx[int(test_split * len(case_idx)):]
            test_case_idx = case_idx[:int(test_split * len(case_idx))]
        else:
            train_case_idx = case_idx[:-int(test_split * len(case_idx))]
            test_case_idx = case_idx[-int(test_split * len(case_idx)):]
        train_indices.append(train_case_idx)
        test_indices.append(test_case_idx)
        if case in plot_cases:
            plot_indices.append(case_idx)
            train_plot_indices.append(train_case_idx)
            test_plot_indices.append(test_case_idx)
            train_plot_len.append(len(train_case_idx))
            test_plot_len.append(len(test_case_idx))
    train_idx = np.concatenate(train_indices)
    test_idx = np.concatenate(test_indices)
    idx = np.concatenate((train_idx, test_idx))
    train_plot_idx = np.concatenate(train_plot_indices)
    test_plot_idx = np.concatenate(test_plot_indices)
    
    train_plot_range, test_plot_range = [0], [0]
    for i in range(len(train_plot_len)):
        train_plot_range.append(train_plot_range[i] + train_plot_len[i])
        test_plot_range.append(test_plot_range[i] + test_plot_len[i])

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    energy_nam_train = energy_nam[train_idx]
    energy_nam_test = energy_nam[test_idx]
    nam_inputs_train = nam_inputs[train_idx]
    nam_inputs_test = nam_inputs[test_idx]
    g_nam_train = g[train_idx]
    g_nam_test = g[test_idx]
    X_train_plot = X[train_plot_idx]
    X_test_plot = X[test_plot_idx]
    y_train_plot = y[train_plot_idx]
    y_test_plot = y[test_plot_idx]
    energy_nam_train_plot = energy_nam[train_plot_idx]
    energy_nam_test_plot = energy_nam[test_plot_idx]
    nam_inputs_train_plot = nam_inputs[train_plot_idx]
    nam_inputs_test_plot = nam_inputs[test_plot_idx]
    g_nam_train_plot = g[train_plot_idx]
    g_nam_test_plot = g[test_plot_idx]
    
    nam_inputs_sr = np.zeros((100, nam_inputs.shape[1]))
    g_sr = np.zeros((100, nam_inputs.shape[1]))
    gp_sr = np.zeros((100, nam_inputs.shape[1]))
    gpp_sr = np.zeros((100, nam_inputs.shape[1]))

    g_sr_train = np.zeros_like(g_nam_train)
    g_sr_test = np.zeros_like(g_nam_test)
    gp_sr_train = np.zeros_like(g_nam_train)
    gp_sr_test = np.zeros_like(g_nam_test)
    gpp_sr_train = np.zeros_like(g_nam_train)
    gpp_sr_test = np.zeros_like(g_nam_test)
    g_sr_train_plot = np.zeros_like(g_nam_train_plot)
    g_sr_test_plot = np.zeros_like(g_nam_test_plot)
    gp_sr_train_plot = np.zeros_like(g_nam_train_plot)
    gp_sr_test_plot = np.zeros_like(g_nam_test_plot)
    gpp_sr_train_plot = np.zeros_like(g_nam_train_plot)
    gpp_sr_test_plot = np.zeros_like(g_nam_test_plot)

    model = pysr.PySRRegressor(
        binary_operators=['+', '*'],
        unary_operators=['log'],
        # unary_operators=['exp', 'cosh', 'softplus(x) = log(1 + exp(x))'],
        maxsize=30,
        niterations=10000000,
        populations=15,
        population_size=50,
        ncycles_per_iteration=500,
        elementwise_loss='L2DistLoss()',
        # loss_function=objective,
        model_selection='best',
        nested_constraints={'log': {'log': 0}},
        # nested_constraints={
        #     'exp': {'exp': 0, 'cosh': 0, 'softplus': 0},
        #     'cosh': {'exp': 0, 'cosh': 0, 'softplus': 0},
        #     'softplus': {'exp': 0, 'cosh': 0, 'softplus': 0}
        # },
        timeout_in_seconds=timeout_in_seconds,
        # enable_autodiff=True,
        temp_equation_file=True,
        tempdir=time,
        delete_tempfiles=False,
        # extra_sympy_mappings={'softplus': lambda x: sympy.log(1. + sympy.exp(x))}
    )

    file = open(f'{time}/equations.csv', 'w')
    file.write('x,g_latex,g_eq,gp_eq,gpp_eq,loss\n')
    for k in range(nam_inputs.shape[1]):
        model.fit(
            nam_inputs_train[::fit_step, k].reshape(-1, 1),
            g_nam_train[::fit_step, k],
            variable_names=['x']
        )

        x = sympy.symbols('x')
        g_latex = model.latex(precision=4)
        g_sympy = model.sympy()
        gp_sympy = sympy.diff(g_sympy, x)
        gpp_sympy = sympy.diff(gp_sympy, x)

        g_lamb = sympy.lambdify(x, g_sympy)
        gp_lamb = sympy.lambdify(x, gp_sympy)
        gpp_lamb = sympy.lambdify(x, gpp_sympy)

        nam_inputs_min = nam_inputs[:, k].min()
        nam_inputs_max = nam_inputs[:, k].max()
        nam_inputs_sr[:, k] = np.linspace(
            nam_inputs_min, nam_inputs_max, num=100
        )
        
        g_sr[:, k] = g_lamb(nam_inputs_sr[:, k])
        gp_sr[:, k] = gp_lamb(nam_inputs_sr[:, k])
        gpp_sr[:, k] = gpp_lamb(nam_inputs_sr[:, k])

        g_sr_train[:, k] = g_lamb(nam_inputs_train[:, k])
        g_sr_test[:, k] = g_lamb(nam_inputs_test[:, k])
        gp_sr_train[:, k] = gp_lamb(nam_inputs_train[:, k])
        gp_sr_test[:, k] = gp_lamb(nam_inputs_test[:, k])
        gpp_sr_train[:, k] = gpp_lamb(nam_inputs_train[:, k])
        gpp_sr_test[:, k] = gpp_lamb(nam_inputs_test[:, k])
        g_sr_train_plot[:, k] = g_lamb(nam_inputs_train_plot[:, k])
        g_sr_test_plot[:, k] = g_lamb(nam_inputs_test_plot[:, k])
        gp_sr_train_plot[:, k] = gp_lamb(nam_inputs_train_plot[:, k])
        gp_sr_test_plot[:, k] = gp_lamb(nam_inputs_test_plot[:, k])
        gpp_sr_train_plot[:, k] = gpp_lamb(nam_inputs_train_plot[:, k])
        gpp_sr_test_plot[:, k] = gpp_lamb(nam_inputs_test_plot[:, k])

        pred_diff = g_sr_train[:, k] - g_nam_train[:, k]
        loss = (pred_diff**2).sum() / len(pred_diff)
        write_eq(file, k, g_latex, g_sympy, gp_sympy, gpp_sympy, loss)
    file.close()

    equations = pd.read_csv(f'{time}/equations.csv')
    g_latex = equations['g_latex'].to_numpy()
    g_eq = equations['g_eq'].to_numpy()
    gp_eq = equations['gp_eq'].to_numpy()
    gpp_eq = equations['gpp_eq'].to_numpy()

    bias = energy_nam - g.sum(axis=1)
    energy_eq = construct_eq(func_of, voigt, lin_weight, g_eq, bias[0])
    print(energy_eq)

    energy_sr_train_plot = g_sr_train_plot.sum(axis=1) + bias[0]
    energy_sr_test_plot = g_sr_test_plot.sum(axis=1) + bias[0]
    plot_DPsi(
        func_of,
        X,
        X_train_plot,
        X_test_plot,
        strain_comp,
        y,
        energy_nam_train_plot,
        energy_nam_test_plot,
        energy_sr_train_plot,
        energy_sr_test_plot,
        train_plot_range,
        test_plot_range,
        plot_cases,
        plot_indices,
        plot_dir
    )
    for k in range(nam_inputs.shape[1]):
        plot_Dhatg(
            nam_inputs_train,
            nam_inputs_test,
            nam_inputs_sr,
            g_nam_train,
            g_nam_test,
            g_sr,
            k + 1,
            plot_dir
        )
