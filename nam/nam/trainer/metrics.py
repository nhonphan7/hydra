from abc import abstractmethod
import numpy as np
import sklearn.metrics as sk_metrics
import torch


class Metric:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class Accuracy(Metric):
    def __init__(
        self, 
        input_type: str = None
    ) -> None:
        self._num = 0
        self._denom = 0
        self._input_type = input_type
        self._updated = False

    def update(self, predictions, targets) -> None:
        # TODO: Exception handling/input checking
        predictions, targets = predictions.detach(), targets.detach()
        if self._input_type == 'scores':
            predictions = predictions.round()
        elif self._input_type == 'logits':
            predictions = torch.sigmoid(predictions).round()

        self._num += (predictions * targets).sum().item()
        self._denom += len(predictions)
        self._updated = True
        return

    def compute(self) -> None:
        if not self._updated:
            # TODO: Find appropriate exception
            raise Exception()
        
        return self._num / self._denom

    def reset(self) -> None:
        self._num = self._denom = 0
        return


class AUC(Metric):
    def __init__(
        self 
    ) -> None:
        self._predictions = []
        self._targets = []
        self._updated = False

    @abstractmethod
    def score_function(self, predictions, targets) -> float:
        pass

    def update(self, predictions, targets) -> None:
        # TODO: Exception handling/input checking
        self._predictions.append(predictions)
        self._targets.append(targets)
        self._updated = True
        return

    def compute(self) -> None:
        if not self._updated:
            # TODO: Find appropriate exception
            raise Exception()
        
        predictions = torch.cat(self._predictions).detach().cpu().numpy()
        targets = torch.cat(self._targets).detach().cpu().numpy()
        return self.score_function(predictions, targets)

    def reset(self) -> None:
        self._predictions = []
        self._targets = []
        return


class AUROC(AUC):
    def __init__(
        self 
    ) -> None:
        super(AUROC, self).__init__()
        return

    def score_function(self, predictions, targets) -> float:
        return sk_metrics.roc_auc_score(targets, predictions)


class AveragePrecision(AUC):
    def __init__(
        self 
    ) -> None:
        super(AveragePrecision, self).__init__()
        return

    def score_function(self, predictions, targets) -> float:
        return sk_metrics.average_precision_score(targets, predictions)


class MeanError(Metric):
    def __init__(
        self 
    ) -> None:
        self._sum_of_errors = 0
        self._num_examples = 0
        self._updated = True
        return

    @abstractmethod
    def distance_func(self, predictions, targets) -> float:
        pass

    def update(self, predictions, targets) -> None:
        # TODO: Exception handling/input checking
        predictions = predictions.detach().cpu().numpy() 
        targets = targets.detach().cpu().numpy()
        self._sum_of_errors += self.distance_func(predictions, targets)
        self._num_examples += predictions.shape[0]
        self._updated = True
        return

    def compute(self) -> None:
        if not self._updated:
            # TODO: Find appropriate exception
            raise Exception()
        
        return self._sum_of_errors / self._num_examples

    def reset(self) -> None:
        self._sum_of_errors = 0
        self._num_examples = 0
        return


class MeanSquaredError(MeanError):
    def __init__(
        self 
    ) -> None:
        super(MeanSquaredError, self).__init__()
        return

    def distance_func(self, predictions, targets) -> float:
        return np.sum((predictions - targets) ** 2)


class RootMeanSquaredError(MeanSquaredError):
    def __init__(
        self 
    ) -> None:
        super(MeanSquaredError, self).__init__()
        return

    def distance_func(self, predictions, targets) -> float:
        return super().distance_func(predictions, targets) ** 0.5


class MeanAbsoluteError(MeanError):
    def __init__(
        self 
    ) -> None:
        super(MeanSquaredError, self).__init__()
        return

    def distance_func(self, predictions, targets) -> float:
        return np.sum(np.absolute(predictions - targets))