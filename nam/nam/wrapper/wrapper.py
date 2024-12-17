import os
import random
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import scipy
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler
import torch

from nam.data import NAMDataset
from nam.models import FC, NAM, MultiTaskNAM
from nam.models import get_num_units
from nam.models.saver import Checkpointer
from nam.trainer import Trainer
from nam.trainer.losses import make_penalized_loss_func
from joblib import load
from torch.autograd import grad
import torch.nn.functional as F

class NAMBase:
    def __init__(
        self,
        units_multiplier: int = 2,
        num_basis_functions: int = 64,
        hidden_sizes: list = [64, 32],
        dropout: float = 0.1,
        feature_dropout: float = 0.05, 
        batch_size: int = 1024,
        num_workers: int = 0,
        num_epochs: int = 1000,
        log_dir: str = None,
        val_split: float = 0.15,
        device: str = 'cpu',
        lr: float = 0.02082,
        decay_rate: float = 0.995,
        output_reg: float = 0.2078,
        l2_reg: float = 0.0,
        save_model_frequency: int = 10,
        patience: int = 60,
        monitor_loss: bool = True,
        early_stop_mode: str = 'min',
        loss_func: Callable = None,
        metric: str = None,
        num_learners: int = 1,
        n_jobs: int = None,
        warm_start: bool = False,
        random_state: int = 42,
        func_of: str = 'D',
        scale: bool = True,
        energy: bool = False,
        sobolev: bool = True,
        nam: bool = True,
        proj_ratio: float = 0.0
    ) -> None:
        self.units_multiplier = units_multiplier
        self.num_basis_functions = num_basis_functions
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.log_dir = log_dir
        self.val_split = val_split
        self.device = device
        self.lr = lr
        self.decay_rate = decay_rate
        self.output_reg = output_reg
        self.l2_reg = l2_reg
        self.save_model_frequency = save_model_frequency
        self.patience = patience
        self.monitor_loss = monitor_loss
        self.early_stop_mode = early_stop_mode
        self.loss_func = loss_func
        self.metric = metric
        self.num_learners = num_learners
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.random_state = random_state
        self.func_of = func_of
        self.scale = scale
        self.energy = energy
        self.sobolev = sobolev
        self.nam = nam
        self.proj_ratio = proj_ratio

        self._best_checkpoint_suffix = 'best'
        self._fitted = False

    def _set_random_state(self):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        return
    
    def _initialize_models(self, X, y):
        # self.num_tasks = y.shape[1] if len(y.shape) > 1 else 1
        self.num_tasks = 1
        self.num_inputs = X.shape[1]
        self.models = []
        self.proj_sizes = round(self.proj_ratio * self.num_inputs) if self.proj_ratio > 0 else self.num_inputs
        for _ in range(self.num_learners):
            if self.nam:
                model = NAM(num_inputs=self.num_inputs,
                    num_units=get_num_units(self.units_multiplier, self.num_basis_functions, X),
                    dropout=self.dropout,
                    feature_dropout=self.feature_dropout,
                    hidden_sizes=self.hidden_sizes,
                    proj_ratio=self.proj_ratio,
                    proj_sizes=self.proj_sizes)
            else:
                model = FC(num_inputs=self.num_inputs,
                    num_units=get_num_units(self.units_multiplier, self.num_basis_functions, X),
                    dropout=self.dropout,
                    feature_dropout=self.feature_dropout,
                    hidden_sizes=self.hidden_sizes)
            self.models.append(model)

        return

    def _models_to_device(self, device):
        for model in self.models:
            model.to(device)

        return
    
    def _initialize_variables(self):
        if self.scale:
            X_scaler = load('scales/X_scaler.pkl')
            self.X_scale_ = torch.tensor(X_scaler.scale_, dtype=torch.float)
            self.X_min_ = torch.tensor(X_scaler.min_, dtype=torch.float)
            y_scaler = load('scales/y_scaler.pkl')
            self.y_scale_ = torch.tensor(y_scaler.scale_, dtype=torch.float)
            self.y_min_ = torch.tensor(y_scaler.min_, dtype=torch.float)
        if self.func_of == 'D' and not self.energy:
            self.X_sf = torch.tensor([[0., 0., 0., 0., 0., 0.]], requires_grad=True)
            if self.scale:
                self.X_sf = self.X_sf * self.X_scale_ + self.X_min_
        return

    def fit(self, X, y, w=None):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            w = w.to_numpy()

        self._set_random_state()
        if not self.warm_start or not self._fitted:
            self._initialize_models(X, y)

        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, w=None) -> None:
        self._models_to_device(self.device)
        
        # self._preprocessor = MinMaxScaler(feature_range = (-1, 1))

        # dataset = NAMDataset(self._preprocessor.fit_transform(X), y, w)
        dataset = NAMDataset(X, y, w)

        self.criterion = make_penalized_loss_func(self.loss_func, 
            self.regression, self.output_reg, self.l2_reg)

        self.trainer = Trainer(
            models=self.models,
            dataset=dataset,
            metric=self.metric,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            num_epochs=self.num_epochs,
            log_dir=self.log_dir,
            val_split=self.val_split,
            test_split=None,
            device=self.device,
            lr=self.lr,
            decay_rate=self.decay_rate,
            save_model_frequency=self.save_model_frequency,
            patience=self.patience,
            monitor_loss=self.monitor_loss,
            early_stop_mode=self.early_stop_mode,
            criterion=self.criterion,
            regression=self.regression,
            num_learners=self.num_learners,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            func_of=self.func_of,
            scale=self.scale,
            energy=self.energy,
            sobolev=self.sobolev,
            nam=self.nam
        )
        
        self.trainer.train_ensemble()
        self.trainer.close()

        # Move models to cpu so predictions can be made on cpu data
        self._models_to_device('cpu')

        self._fitted = True
        return self

    def predict(self, X) -> ArrayLike:
        if not self._fitted:
            raise NotFittedError('''This NAM instance is not fitted yet. Call \'fit\' 
                with appropriate arguments before using this method.''')

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        # X = self._preprocessor.transform(X)
        X = torch.tensor(X, requires_grad=True, dtype=torch.float)
        predictions = np.zeros((X.shape[0],))
        if self.num_tasks > 1:
            predictions = np.zeros((X.shape[0], self.num_tasks))

        for model in self.models:
            preds, _ = model.forward(X)
            predictions += preds.detach().cpu().numpy()

        # predictions = self._preprocessor.inverse_transform(predictions)
        return predictions / self.num_learners

    def plot(self, feature_index) -> None:
        num_samples = 1000
        X = np.zeros((num_samples, self.num_inputs))
        X[:, feature_index] = np.linspace(-1.0, 1.0, num_samples)
        
        feature_outputs = []
        for model in self.models:
            # (examples, tasks, features)
            _, fnns_out = model.forward(torch.tensor(X, dtype=torch.float32))
            if self.num_tasks == 1:
                fnns_out = fnns_out.unsqueeze(dim=1)
            # (examples, tasks)
            feature_outputs.append(fnns_out[:, :, feature_index].detach().cpu().numpy())

        # (learners, examples, tasks)
        feature_outputs = np.stack(feature_outputs, axis=0)
        # (examples, tasks)
        y = np.mean(feature_outputs, axis=0).squeeze()
        conf_int = np.std(feature_outputs, axis=0).squeeze()
        # TODO: Scale conf_int according to units of y

        # X = self._preprocessor.inverse_transform(X)
        
        return {'x': X[:, feature_index], 'y': y, 'conf_int': conf_int}

    def load_checkpoints(self, checkpoint_dir):
        self.models = []
        for i in range(self.num_learners):
            checkpointer = Checkpointer(os.path.join(checkpoint_dir, str(i)))
            model = checkpointer.load(self._best_checkpoint_suffix)
            model.eval()
            self.num_tasks = 1 if isinstance(model, FC) or isinstance(model, NAM) else model.num_tasks
            self.models.append(model)

        self._fitted = True
        return
    
    def predict_plot(self, X, y) -> ArrayLike:
        if not self._fitted:
            raise NotFittedError('''This NAM instance is not fitted yet. Call \'fit\' 
                with appropriate arguments before using this method.''')
        self._initialize_variables()

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()
        # X = self._preprocessor.transform(X)
        X = torch.tensor(X, requires_grad=True, dtype=torch.float)
        y = torch.tensor(y, requires_grad=True, dtype=torch.float)
        proj_sizes = round(self.proj_ratio * X.shape[1]) if self.proj_ratio > 0 else X.shape[1]
        nam_inputs = np.zeros((X.shape[0], proj_sizes))
        predictions = np.zeros((X.shape[0],))
        predictions_X = np.zeros((X.shape[0], X.shape[1]))
        predictions_nam = np.zeros((X.shape[0], proj_sizes))
        S = np.zeros((X.shape[0], 6))
        if self.func_of == 'D' and not self.energy:
            X = torch.cat((X, self.X_sf))

        if self.num_tasks == 1 and self.num_learners == 1:
            feature_outputs = []
            lin_weight = None
            for model in self.models:
                # (examples, features)
                if self.nam:
                    preds, fnns_out, nam_in = model.forward(X)
                    if self.proj_ratio > 0:
                        lin_weight = model.linear.weight.detach().cpu().numpy()
                        if self.sobolev:
                            preds_nam = grad(preds.sum(), nam_in, create_graph=True)[0]
                else:
                    preds = model.forward(X)
                
                if self.sobolev:
                    preds_X = grad(preds.sum(), X, create_graph=True)[0]
                    if self.func_of == 'D' and self.scale:
                        if self.energy:
                            S_unscale = preds_X * self.X_scale_ / self.y_scale_[0]
                            S_scale = S_unscale * self.y_scale_[-6:] + self.y_min_[-6:]
                            energy_loss = F.mse_loss(preds, y[:, 0])
                            stress_loss = F.mse_loss(S_scale, y[:, -6:])
                        else:
                            S_unscale = preds_X[:-1] * self.X_scale_
                            S_scale = S_unscale * self.y_scale_ + self.y_min_
                            energy_loss = F.mse_loss(preds[-1].unsqueeze(0), torch.zeros(1))
                            stress_loss = F.mse_loss(S_scale, y)
                        print(f'Energy Loss: {energy_loss:.10f}')
                        print(f'Stress Loss: {stress_loss:.10f}')
                else:
                    energy_loss = F.mse_loss(preds, y)
                    print(f'Energy Loss: {energy_loss:.10f}')
                
                if self.energy:
                    predictions += preds.detach().cpu().numpy()
                    if self.nam:
                        feature_outputs.append(fnns_out.detach().cpu().numpy())
                        if self.proj_ratio > 0:
                            nam_inputs += nam_in.detach().cpu().numpy()
                            if self.sobolev:
                                predictions_nam += preds_nam.detach().cpu().numpy()
                    if self.sobolev:
                        predictions_X += preds_X.detach().cpu().numpy()
                        S += S_scale.detach().cpu().numpy()
                else:
                    predictions += preds[:-1].detach().cpu().numpy()
                    if self.nam:
                        feature_outputs.append(fnns_out[:-1].detach().cpu().numpy())
                        if self.proj_ratio > 0:
                            nam_inputs += nam_in[:-1].detach().cpu().numpy()
                            if self.sobolev:
                                predictions_nam += preds_nam[:-1].detach().cpu().numpy()
                    if self.sobolev:
                        predictions_X += preds_X[:-1].detach().cpu().numpy()
                        S += S_scale.detach().cpu().numpy()
            
            # (examples, features)
            if self.nam:
                feature_outputs = np.concatenate(feature_outputs)

        # predictions = self._preprocessor.inverse_transform(predictions)
        return predictions, predictions_X, S, feature_outputs, nam_inputs, predictions_nam, lin_weight


class NAMClassifier(NAMBase):
    def __init__(
        self,
        units_multiplier: int = 2,
        num_basis_functions: int = 64,
        hidden_sizes: list = [64, 32],
        dropout: float = 0.1,
        feature_dropout: float = 0.05, 
        batch_size: int = 1024,
        num_workers: int = 0,
        num_epochs: int = 1000,
        log_dir: str = None,
        val_split: float = 0.15,
        device: str = 'cpu',
        lr: float = 0.02082,
        decay_rate: float = 0.0,
        output_reg: float = 0.2078,
        l2_reg: float = 0.0,
        save_model_frequency: int = 10,
        patience: int = 60,
        monitor_loss: bool = True,
        early_stop_mode: str = 'min',
        loss_func: Callable = None,
        metric: str = None,
        num_learners: int = 1,
        n_jobs: int = None,
        warm_start: bool = False,
        random_state: int = 42
    ) -> None:
        super(NAMClassifier, self).__init__(
            units_multiplier=units_multiplier,
            num_basis_functions=num_basis_functions,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            feature_dropout=feature_dropout,
            batch_size=batch_size,
            num_workers=num_workers,
            num_epochs=num_epochs,
            log_dir=log_dir,
            val_split=val_split,
            device=device,
            lr=lr,
            decay_rate=decay_rate,
            output_reg=output_reg,
            l2_reg=l2_reg,
            save_model_frequency=save_model_frequency,
            patience=patience,
            monitor_loss=monitor_loss,
            early_stop_mode=early_stop_mode,
            loss_func=loss_func,
            metric=metric,
            num_learners=num_learners,
            n_jobs=n_jobs,
            warm_start = warm_start,
            random_state=random_state
        )
        self.regression = False

    def fit(self, X, y, w=None):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy()
        if isinstance(y, (pd.DataFrame, pd.Series)):
            w = w.to_numpy()
            
        if len(np.unique(y[~np.isnan(y)])) > 2:
            raise ValueError('More than two unique y-values detected. Multiclass classification not currently supported.')
        return super().fit(X, y, w)

    def predict_proba(self, X) -> ArrayLike:
        out = scipy.special.expit(super().predict(X))
        return out

    def predict(self, X) -> ArrayLike:
        return self.predict_proba(X).round()

    
class NAMRegressor(NAMBase):
    def __init__(
        self,
        units_multiplier: int = 2,
        num_basis_functions: int = 64,
        hidden_sizes: list = [64, 32],
        dropout: float = 0.1,
        feature_dropout: float = 0.05, 
        batch_size: int = 1024,
        num_workers: int = 0,
        num_epochs: int = 1000,
        log_dir: str = None,
        val_split: float = 0.15,
        device: str = 'cpu',
        lr: float = 0.02082,
        decay_rate: float = 0.0,
        output_reg: float = 0.2078,
        l2_reg: float = 0.0,
        save_model_frequency: int = 10,
        patience: int = 60,
        monitor_loss: bool = True,
        early_stop_mode: str = 'min',
        loss_func: Callable = None,
        metric: str = None,
        num_learners: int = 1,
        n_jobs: int = None,
        warm_start: bool = False,
        random_state: int = 42,
        func_of: str = 'D',
        scale: bool = True,
        energy: bool = False,
        sobolev: bool = True,
        nam: bool = True,
        proj_ratio: float = 0.0
    ) -> None:
        super(NAMRegressor, self).__init__(
            units_multiplier=units_multiplier,
            num_basis_functions=num_basis_functions,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            feature_dropout=feature_dropout,
            batch_size=batch_size,
            num_workers=num_workers,
            num_epochs=num_epochs,
            log_dir=log_dir,
            val_split=val_split,
            device=device,
            lr=lr,
            decay_rate=decay_rate,
            output_reg=output_reg,
            l2_reg=l2_reg,
            save_model_frequency=save_model_frequency,
            patience=patience,
            monitor_loss=monitor_loss,
            early_stop_mode=early_stop_mode,
            loss_func=loss_func,
            metric=metric,
            num_learners=num_learners,
            n_jobs=n_jobs,
            warm_start = warm_start,
            random_state=random_state,
            func_of=func_of,
            scale=scale,
            energy=energy,
            sobolev=sobolev,
            nam=nam,
            proj_ratio=proj_ratio
        )
        self.regression = True


class MultiTaskNAMClassifier(NAMClassifier):
    def __init__(
        self,
        units_multiplier: int = 2,
        num_basis_functions: int = 64,
        hidden_sizes: list = [64, 32],
        num_subnets: int = 2,
        dropout: float = 0.1,
        feature_dropout: float = 0.05, 
        batch_size: int = 1024,
        num_workers: int = 0,
        num_epochs: int = 1000,
        log_dir: str = None,
        val_split: float = 0.15,
        device: str = 'cpu',
        lr: float = 0.02082,
        decay_rate: float = 0.0,
        output_reg: float = 0.2078,
        l2_reg: float = 0.0,
        save_model_frequency: int = 10,
        patience: int = 60,
        monitor_loss: bool = True,
        early_stop_mode: str = 'min',
        loss_func: Callable = None,
        metric: str = None,
        num_learners: int = 1,
        n_jobs: int = None,
        warm_start: bool = False,
        random_state: int = 42
    ) -> None:
        super(MultiTaskNAMClassifier, self).__init__(
            units_multiplier=units_multiplier,
            num_basis_functions=num_basis_functions,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            feature_dropout=feature_dropout,
            batch_size=batch_size,
            num_workers=num_workers,
            num_epochs=num_epochs,
            log_dir=log_dir,
            val_split=val_split,
            device=device,
            lr=lr,
            decay_rate=decay_rate,
            output_reg=output_reg,
            l2_reg=l2_reg,
            save_model_frequency=save_model_frequency,
            patience=patience,
            monitor_loss=monitor_loss,
            early_stop_mode=early_stop_mode,
            loss_func=loss_func,
            metric=metric,
            num_learners=num_learners,
            n_jobs=n_jobs,
            warm_start = warm_start,
            random_state=random_state
        )
        self.num_subnets = num_subnets

    def _initialize_models(self, X, y):
        self.num_inputs = X.shape[1]
        self.num_tasks = y.shape[1] if len(y.shape) > 1 else 1
        self.models = []
        for _ in range(self.num_learners):
            model = MultiTaskNAM(num_inputs=X.shape[1],
                num_units=get_num_units(self.units_multiplier, self.num_basis_functions, X),
                num_subnets=self.num_subnets,
                num_tasks=y.shape[1],
                dropout=self.dropout,
                feature_dropout=self.feature_dropout,
                hidden_sizes=self.hidden_sizes)
            model.to(self.device)
            self.models.append(model)


class MultiTaskNAMRegressor(NAMRegressor):
    def __init__(
        self,
        units_multiplier: int = 2,
        num_basis_functions: int = 64,
        hidden_sizes: list = [64, 32],
        num_subnets: int = 2,
        dropout: float = 0.1,
        feature_dropout: float = 0.05, 
        batch_size: int = 1024,
        num_workers: int = 0,
        num_epochs: int = 1000,
        log_dir: str = None,
        val_split: float = 0.15,
        device: str = 'cpu',
        lr: float = 0.02082,
        decay_rate: float = 0.995,
        output_reg: float = 0.2078,
        l2_reg: float = 0.0,
        save_model_frequency: int = 10,
        patience: int = 60,
        monitor_loss: bool = True,
        early_stop_mode: str = 'min',
        loss_func: Callable = None,
        metric: str = None,
        num_learners: int = 1,
        n_jobs: int = None,
        warm_start: bool = False,
        random_state: int = 42
    ) -> None:
        super(MultiTaskNAMRegressor, self).__init__(
            units_multiplier=units_multiplier,
            num_basis_functions=num_basis_functions,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            feature_dropout=feature_dropout,
            batch_size=batch_size,
            num_workers=num_workers,
            num_epochs=num_epochs,
            log_dir=log_dir,
            val_split=val_split,
            device=device,
            lr=lr,
            decay_rate=decay_rate,
            output_reg=output_reg,
            l2_reg=l2_reg,
            save_model_frequency=save_model_frequency,
            patience=patience,
            monitor_loss=monitor_loss,
            early_stop_mode=early_stop_mode,
            loss_func=loss_func,
            metric=metric,
            num_learners=num_learners,
            n_jobs=n_jobs,
            warm_start = warm_start,
            random_state=random_state
        )
        self.num_subnets = num_subnets

    def _initialize_models(self, X, y):
        self.num_inputs = X.shape[1]
        self.num_tasks = y.shape[1] if len(y.shape) > 1 else 1
        self.models = []
        for _ in range(self.num_learners):
            model = MultiTaskNAM(num_inputs=X.shape[1],
                num_units=get_num_units(self.units_multiplier, self.num_basis_functions, X),
                num_subnets=self.num_subnets,
                num_tasks=y.shape[1],
                dropout=self.dropout,
                feature_dropout=self.feature_dropout,
                hidden_sizes=self.hidden_sizes)
            model.to(self.device)
            self.models.append(model)