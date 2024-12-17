import datetime
import joblib
import os
import sympy
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(1, '../')
from nam.wrapper import NAMRegressor
from utils import *


if __name__ == '__main__':
    # Synthetic cubic data
    # case 0: uniaxial compression (11 direction)
    # case 1: uniaxial compression (22 direction)
    # case 2: positive simple shear (12 direction)
    # case 3: positive simple shear (13 direction)
    # case 4: negative simple shear (13 direction)
    # case 5: negative simple shear (12 direction)
    # case 6: uniaxial tension (11 direction)
    # case 7: uniaxial tension (22 direction)
    # case 8: equibiaxial compression (11 and 22 directions)
    # case 9: equibiaxial compression (22 and 33 directions)

    noise_factor = 0.
    if noise_factor > 0:
        data_dir = f'../../cubic/cubic_noise{int(noise_factor * 100)}.csv'
    else:
        data_dir = '../../cubic/cubic.csv'
    plot_cases = [0, 6]
    strain_comp = 11
    save_df = True
    
    test_split = 0.15
    num_epochs = 1000
    func_of = 'D'
    scale = True
    energy = True
    sobolev = True
    nam = True
    proj_ratio = 1.5

    small_size = 20
    large_size = 24
    plt.rc('font', size=small_size)
    plt.rc('axes', labelsize=large_size)

    # time = datetime.datetime.now().isoformat()
    time = '2024-07-12T02:04:28.590420'
    plot_dir = f'{time}/figures'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if scale and not os.path.exists('scales'):
        os.makedirs('scales')

    voigt = [11, 22, 33, 23, 13, 12]
    df = pd.read_csv(data_dir)
    X = df[['d11', 'd22', 'd33', 'd23', 'd13', 'd12']].to_numpy()
    y = df[['psi', 'sd11', 'sd22', 'sd33', 'sd23', 'sd13', 'sd12']].to_numpy()
    if scale:
        X_scaler = MinMaxScaler()
        X_scale = X_scaler.fit_transform(X)
        if noise_factor > 0:
            joblib.dump(
                X_scaler, f'scales/X_scaler_noise{int(noise_factor * 100)}.pkl'
            )
        else:
            joblib.dump(X_scaler, 'scales/X_scaler.pkl')
        y_scaler = MinMaxScaler()
        y_scale = y_scaler.fit_transform(y)
        if noise_factor > 0:
            joblib.dump(
                y_scaler, f'scales/y_scaler_noise{int(noise_factor * 100)}.pkl'
            )
        else:
            joblib.dump(y_scaler, 'scales/y_scaler.pkl')

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
    
    if scale:
        X_train = X_scale[train_idx]
        X_test = X_scale[test_idx]
        y_train = y_scale[train_idx]
        y_test = y_scale[test_idx]
        X_train_plot = X_scale[train_plot_idx]
        X_test_plot = X_scale[test_plot_idx]
        y_train_plot = y_scale[train_plot_idx]
        y_test_plot = y_scale[test_plot_idx]
    else:
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        X_train_plot = X[train_plot_idx]
        X_test_plot = X[test_plot_idx]
        y_train_plot = y[train_plot_idx]
        y_test_plot = y[test_plot_idx]
    
    model = NAMRegressor(
        num_basis_functions=128,
        hidden_sizes=[64, 32],
        dropout=0.,
        feature_dropout=0.,
        batch_size=64,
        num_epochs=num_epochs,
        log_dir=f'{time}/output',
        val_split=0.15,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        lr=0.001,
        decay_rate=0.,
        output_reg=0.,
        l2_reg=0.,
        patience=num_epochs,
        num_learners=1,
        n_jobs=1,
        random_state=42,
        func_of=func_of,
        scale=scale,
        energy=energy,
        sobolev=sobolev,
        nam=nam,
        proj_ratio=proj_ratio
    )
    # model.fit(X_train, y_train)
    model.load_checkpoints(f'{time}/output')
    
    (
        preds_train,
        preds_X_train,
        stress_train,
        feat_outputs_train,
        nam_inputs_train,
        preds_nam_train,
        lin_weight
    ) = model.predict_plot(X_train, y_train)
    (
        preds_test,
        preds_X_test,
        stress_test,
        feat_outputs_test,
        nam_inputs_test,
        preds_nam_test,
        _
    ) = model.predict_plot(X_test, y_test)

    nam_inputs = np.concatenate((nam_inputs_train, nam_inputs_test))
    preds = np.concatenate((preds_train, preds_test))
    preds_X = np.concatenate((preds_X_train, preds_X_test))
    preds_nam = np.concatenate((preds_nam_train, preds_nam_test))
    stress = np.concatenate((stress_train, stress_test))

    if nam and proj_ratio > 0:
        feat_outputs = np.concatenate((feat_outputs_train, feat_outputs_test))
        if noise_factor > 0:
            np.save(
                f'lin_weight_{func_of}{nam_inputs.shape[1]}'
                f'_noise{int(noise_factor * 100)}.npy',
                lin_weight)
        else:
            np.save(f'lin_weight_{func_of}{nam_inputs.shape[1]}.npy', lin_weight)
        if save_df:
            if scale:
                y_pred = np.hstack((preds.reshape(-1, 1), stress))
                y_pred = y_scaler.inverse_transform(y_pred)
                preds = y_pred[:, 0]
                stress = y_pred[:, -6:]
            df['psi_nam'] = np.zeros(df.shape[0])
            for comp in voigt:
                df[f'sd{comp}_nam'] = np.zeros(df.shape[0])
            for k in range(nam_inputs.shape[1]):
                df[f'g{k + 1}'] = np.zeros(df.shape[0])
            for k in range(nam_inputs.shape[1]):
                df[f'gp{k + 1}'] = np.zeros(df.shape[0])
            for i, j in enumerate(idx):
                df.loc[j, 'psi_nam'] = preds[i]
                for comp in voigt:
                    df.loc[j, f'sd{comp}_nam'] = stress[i, comp2idx(comp)]
                for k in range(nam_inputs.shape[1]):
                    df.loc[j, f'g{k + 1}'] = feat_outputs[i, k]
                for k in range(nam_inputs.shape[1]):
                    df.loc[j, f'gp{k + 1}'] = preds_nam[i, k]
            if noise_factor > 0:
                df.to_csv(
                    f'cubic_{func_of}{nam_inputs.shape[1]}'
                    f'_noise{int(noise_factor * 100)}.csv',
                    index=False
                )
            else:
                df.to_csv(
                    f'cubic_{func_of}{nam_inputs.shape[1]}.csv', index=False
                )

    (
        preds_train_plot,
        preds_X_train_plot,
        stress_train_plot,
        feat_outputs_train_plot,
        nam_inputs_train_plot,
        preds_nam_train_plot,
        _
    ) = model.predict_plot(X_train_plot, y_train_plot)
    (
        preds_test_plot,
        preds_X_test_plot,
        stress_test_plot,
        feat_outputs_test_plot,
        nam_inputs_test_plot,
        preds_nam_test_plot,
        _
    ) = model.predict_plot(X_test_plot, y_test_plot)
    
    if scale:
        X_train_plot = X_scaler.inverse_transform(X_train_plot)
        X_test_plot = X_scaler.inverse_transform(X_test_plot)
        y_train_plot_pred = np.hstack(
            (preds_train_plot.reshape(-1, 1), stress_train_plot)
        )
        y_train_plot_pred = y_scaler.inverse_transform(y_train_plot_pred)
        preds_train_plot = y_train_plot_pred[:, 0]
        stress_train_plot = y_train_plot_pred[:, -6:]
        y_test_plot_pred = np.hstack(
            (preds_test_plot.reshape(-1, 1), stress_test_plot)
        )
        y_test_plot_pred = y_scaler.inverse_transform(y_test_plot_pred)
        preds_test_plot = y_test_plot_pred[:, 0]
        stress_test_plot = y_test_plot_pred[:, -6:]
    
    plot_DPsi(
        func_of,
        X,
        X_train_plot,
        X_test_plot,
        strain_comp,
        y[:, 0],
        preds_train_plot,
        preds_test_plot,
        train_plot_range,
        test_plot_range,
        plot_cases,
        plot_indices,
        plot_dir
    )
    for comp in voigt:
        plot_DSDcomp(
            X,
            X_train_plot,
            X_test_plot,
            strain_comp,
            y[:, -6:],
            stress_train_plot,
            stress_test_plot,
            comp,
            train_plot_range,
            test_plot_range,
            plot_cases,
            plot_indices,
            plot_dir
        )
    if nam and proj_ratio > 0:
        U, S, Vh = np.linalg.svd(lin_weight, full_matrices=False)
        M = sympy.Matrix(lin_weight)
        np.set_printoptions(precision=3, suppress=True)
        print(lin_weight)
        print(S)
        print(M.nullspace())
        print(M.rank())
        print(M.rref())
        for k in range(nam_inputs.shape[1]):
            plot_Dhatg(
                nam_inputs_train,
                nam_inputs_test,
                feat_outputs_train,
                feat_outputs_test,
                k + 1,
                plot_dir
            )
            plot_Dhatgp(
                nam_inputs_train,
                nam_inputs_test,
                preds_nam_train,
                preds_nam_test,
                k + 1,
                plot_dir
            )
    plot_loss(time, plot_dir)
