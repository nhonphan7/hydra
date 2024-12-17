from typing import Sequence
from typing import Tuple

import torch
import torch.nn as nn

from nam.models.featurenn import FeatureNN, MultiFeatureNN


class FC(torch.nn.Module):

    def __init__(
        self,
        num_inputs: int,
        num_units: list,
        hidden_sizes: list,
        dropout: float,
        feature_dropout: float
    ) -> None:
        super(FC, self).__init__()
        assert len(num_units) == num_inputs
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.feature_dropout = feature_dropout

        self.dropout_layer = nn.Dropout(p=self.feature_dropout)

        ## Builds the FeatureNNs on the first call.
        self.feature_nns = FeatureNN(
            input_shape=num_inputs, 
            num_units=self.num_units[0], 
            dropout=self.dropout, feature_num=0, 
            hidden_sizes=self.hidden_sizes
        )

        self._bias = torch.nn.Parameter(data=torch.zeros(1))

    def calc_outputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Returns the output computed by each feature net."""
        return self.feature_nns(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.calc_outputs(inputs).squeeze()
        return out + self._bias


class NAM(torch.nn.Module):

    def __init__(
        self,
        num_inputs: int,
        num_units: list,
        hidden_sizes: list,
        dropout: float,
        feature_dropout: float,
        proj_ratio: float,
        proj_sizes: int
    ) -> None:
        super(NAM, self).__init__()
        assert len(num_units) == num_inputs
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.proj_ratio = proj_ratio
        self.proj_sizes = proj_sizes

        self.dropout_layer = nn.Dropout(p=self.feature_dropout)

        if self.proj_ratio > 0:
            self.linear = nn.Linear(
                in_features=self.num_inputs,
                out_features=self.proj_sizes,
                bias=False
            )
        
        ## Builds the FeatureNNs on the first call.
        self.feature_nns = nn.ModuleList([
            FeatureNN(
                input_shape=1, 
                num_units=self.num_units[0], 
                dropout=self.dropout, feature_num=i, 
                hidden_sizes=self.hidden_sizes
            )
            for i in range(self.proj_sizes)
        ])

        self._bias = torch.nn.Parameter(data=torch.zeros(1))

    def calc_outputs(self, inputs: torch.Tensor) -> Tuple[Sequence[torch.Tensor], torch.Tensor]:
        """Returns the output computed by each feature net."""
        nam_inputs = inputs
        if self.proj_ratio > 0:
            nam_inputs = self.linear(inputs)
        return [self.feature_nns[i](nam_inputs[:, i]) for i in range(self.proj_sizes)], nam_inputs

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        individual_outputs, nam_inputs = self.calc_outputs(inputs)
        conc_out = torch.cat(individual_outputs, dim=-1)
        dropout_out = self.dropout_layer(conc_out)

        out = torch.sum(dropout_out, dim=-1)
        return out + self._bias, dropout_out, nam_inputs


class MultiTaskNAM(torch.nn.Module):

    def __init__(
        self,
        num_inputs: list,
        num_units: int,
        num_subnets: int,
        num_tasks: int,
        hidden_sizes: list,
        dropout: float,
        feature_dropout: float
    ) -> None:
        super(MultiTaskNAM, self).__init__()

        assert len(num_units) == num_inputs
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.num_subnets = num_subnets
        self.num_tasks = num_tasks
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.feature_dropout = feature_dropout

        self.dropout_layer = nn.Dropout(p=self.feature_dropout)

        ## Builds the FeatureNNs on the first call.
        self.feature_nns = nn.ModuleList([
            MultiFeatureNN(
                    input_shape=1,
                    feature_num=i,
                    num_units=self.num_units[i],
                    num_subnets=self.num_subnets,
                    num_tasks=self.num_tasks,
                    dropout=self.dropout,
                    hidden_sizes=self.hidden_sizes
                )
            for i in range(self.num_inputs)
        ])
        
        self._bias = torch.nn.Parameter(data=torch.zeros(1, self.num_tasks))

    def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """Returns the output computed by each feature net."""
        return [self.feature_nns[i](inputs[:, i]) for i in range(self.num_inputs)]

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # tuple: (batch, num_tasks) x num_inputs
        individual_outputs = self.calc_outputs(inputs)
        # (batch, num_tasks, num_inputs)
        stacked_out = torch.stack(individual_outputs, dim=-1).squeeze(dim=1)
        dropout_out = self.dropout_layer(stacked_out)

        # (batch, num_tasks)
        summed_out = torch.sum(dropout_out, dim=2) + self._bias
        return summed_out, dropout_out

    def feature_output(self, feature_index, inputs):
        return self.feature_nns[feature_index](inputs)
