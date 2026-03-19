"""Sklearn-compatible predictors backed by the hierarchical TabICL model."""

import numpy as np
import torch

from .model import TabICLModel
from .train import load_checkpoint


__all__ = ['HierarchicalTabICLClassifier', 'HierarchicalTabICLRegressor']


class HierarchicalTabICLClassifier:
    """Sklearn-compatible classifier using the hierarchical TabICL model.

    Ensemble prediction is achieved by averaging logits over ``n_estimators``
    random feature permutations of the context set.

    Parameters
    ----------
    checkpoint : str or None
        Path to a saved checkpoint.  If None, the model uses random weights
        (useful for testing or further fine-tuning).
    device : str
        ``'auto'``, ``'cuda'``, or ``'cpu'``.
    n_estimators : int
        Number of feature-permutation ensemble members.
    **model_kwargs :
        Overrides for ``TabICLModel`` constructor arguments (ignored when a
        checkpoint is supplied, since the config is read from the checkpoint).
    """

    def __init__(self, checkpoint=None, *, device='auto', n_estimators=4,
                 **model_kwargs):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device       = device
        self.n_estimators = n_estimators

        if checkpoint:
            self.model, _ = load_checkpoint(checkpoint, device=device)
        else:
            from .model import create_model
            self.model = create_model(**model_kwargs).to(device)
        self.model.eval()

    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Store training data.  No gradient updates are performed.

        Parameters
        ----------
        X : array-like [n_train, n_features]
        y : array-like [n_train]  — class labels (any hashable type)

        Returns
        -------
        self
        """
        self.X_train_   = np.asarray(X, dtype=np.float32)
        y_arr           = np.asarray(y)
        self.classes_   = np.unique(y_arr)
        enc             = {c: i for i, c in enumerate(self.classes_)}
        self.y_encoded_ = np.array([enc[c] for c in y_arr], dtype=np.int64)
        self._n_feat    = self.X_train_.shape[1]
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        """Return class probabilities [n_test, n_classes].

        Averages softmax probabilities over ``n_estimators`` random feature
        permutations of the training context.
        """
        X_test    = np.asarray(X, dtype=np.float32)
        n_test    = X_test.shape[0]
        n_classes = len(self.classes_)
        rng       = np.random.default_rng(42)
        logits_acc = np.zeros((n_test, n_classes), dtype=np.float64)

        for _ in range(self.n_estimators):
            perm  = rng.permutation(self._n_feat)
            X_ctx = torch.tensor(self.X_train_[:, perm][np.newaxis],
                                 dtype=torch.float32, device=self.device)
            y_ctx = torch.tensor(self.y_encoded_[np.newaxis], device=self.device)
            X_qry = torch.tensor(X_test[:, perm][np.newaxis],
                                 dtype=torch.float32, device=self.device)
            with torch.no_grad():
                logits = self.model(X_ctx, y_ctx, X_qry)   # [1, n_test, max_cls]
            logits_acc += logits[0, :, :n_classes].cpu().float().numpy()

        logits_acc /= self.n_estimators
        logits_acc -= logits_acc.max(1, keepdims=True)
        exp_l = np.exp(logits_acc)
        return exp_l / exp_l.sum(1, keepdims=True)

    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        """Return predicted class labels."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y) -> float:
        """Return accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y)))


class HierarchicalTabICLRegressor:
    """Regression wrapper using quantile-binned targets.

    The model is trained (or expected to have been trained) with
    ``max_classes`` = ``n_bins`` output classes that correspond to quantile
    bins of the training target.  At prediction time the expected bin midpoint
    is returned.

    Parameters
    ----------
    checkpoint : str or None
    device : str
    n_estimators : int
    n_bins : int
        Number of quantile bins (should match the ``max_classes`` used during
        training).
    **model_kwargs :
        Passed to ``TabICLModel`` when no checkpoint is given.
    """

    def __init__(self, checkpoint=None, *, device='auto', n_estimators=4,
                 n_bins=50, **model_kwargs):
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device       = device
        self.n_estimators = n_estimators
        self.n_bins       = n_bins

        if checkpoint:
            self.model, _ = load_checkpoint(checkpoint, device=device)
        else:
            from .model import create_model
            model_kwargs.setdefault('max_classes', n_bins)
            self.model = create_model(**model_kwargs).to(device)
        self.model.eval()

    # ------------------------------------------------------------------
    def fit(self, X, y):
        """Fit the regressor (quantise targets; no gradient updates)."""
        self.X_train_ = np.asarray(X, dtype=np.float32)
        y_raw         = np.asarray(y, dtype=np.float64)
        self._y_mean  = y_raw.mean()
        self._y_std   = y_raw.std() + 1e-8
        y_norm        = (y_raw - self._y_mean) / self._y_std

        edges          = np.linspace(0, 100, self.n_bins + 1)
        thresholds     = np.percentile(y_norm, edges[1:-1])
        self.y_bins_   = np.digitize(y_norm, thresholds).clip(0, self.n_bins - 1).astype(np.int64)
        self._thresh   = thresholds
        self._n_feat   = self.X_train_.shape[1]
        return self

    # ------------------------------------------------------------------
    def predict(self, X) -> np.ndarray:
        """Return predicted target values."""
        X_test     = np.asarray(X, dtype=np.float32)
        rng        = np.random.default_rng(42)
        acc        = np.zeros(len(X_test), dtype=np.float64)

        # Bin midpoints in normalised space
        edges_norm = np.concatenate([[-4.0], self._thresh, [4.0]])
        mids       = (edges_norm[:-1] + edges_norm[1:]) / 2.0

        for _ in range(self.n_estimators):
            perm  = rng.permutation(self._n_feat)
            X_ctx = torch.tensor(self.X_train_[:, perm][np.newaxis],
                                 dtype=torch.float32, device=self.device)
            y_ctx = torch.tensor(self.y_bins_[np.newaxis], device=self.device)
            X_qry = torch.tensor(X_test[:, perm][np.newaxis],
                                 dtype=torch.float32, device=self.device)
            with torch.no_grad():
                logits = self.model(X_ctx, y_ctx, X_qry)   # [1, n_test, n_bins]
            probs = logits[0].softmax(-1).cpu().float().numpy()
            acc  += probs @ mids

        acc /= self.n_estimators
        return acc * self._y_std + self._y_mean

    def score(self, X, y) -> float:
        """Return R² score."""
        from sklearn.metrics import r2_score
        return r2_score(np.asarray(y), self.predict(X))
