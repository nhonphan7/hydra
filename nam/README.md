# Neural Additive Models for InterpretML

![PyPI Python Version](https://img.shields.io/pypi/pyversions/nam)
![GitHub license](https://img.shields.io/github/license/lemeln/nam)

*This repo is forked from AmrMKayid/nam to be adapted for integration of NAMs and Multitask NAMs into InterpretML.*

Neural Additive Models (NAMs) are an interpretable ("glassbox") machine learning technique that jointly learns a separate network on each input feature. The technique is described in [Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://proceedings.neurips.cc/paper/2021/file/251bd0442dfcc53b5a761e050f8022b8-Paper.pdf).

Please see https://github.com/google-research/google-research/tree/master/neural_additive_models for the original TensorFlow implementation of NAMs used in the paper.

## Example Usage

```python
from nam.wrapper import NAMClassifier
import sklearn.metrics as sk_metrics

model = NAMClassifier(
            num_epochs=1000,
            num_learners=20,
            metric='auroc',
            early_stop_mode='max',
            monitor_loss=False,
            n_jobs=10,
            random_state=random_state
        )

model.fit(X_train, y_train)
pred = model.predict_proba(X_test)
sk_metrics.roc_auc_score(y_test, pred)
```
See '''examples.ipynb''' for more details.

## Acknowledgements

This repo, which was created by Levi Melnick, forks AmrMKayid/nam, which is maintained by Amr Kayid and Nicholas Frosst. The code extends and builds on top of their PyTorch implementation.

```bibtex
@article{agarwal2021neural,
  title={Neural additive models: Interpretable machine learning with neural nets},
  author={Agarwal, Rishabh and Melnick, Levi and Frosst, Nicholas and Zhang, Xuezhou and Lengerich, Ben and Caruana, Rich and Hinton, Geoffrey E},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
