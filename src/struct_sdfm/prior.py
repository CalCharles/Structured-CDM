"""Hierarchical LCS synthetic data prior.

Generates tabular datasets via a hierarchical composition of Local Causal
Structures (LCS). Each LCS is a small topologically-ordered DAG (3-8 nodes)
with random edge functions. LCS blocks are composed hierarchically: nodes in
later blocks can depend on nodes in any earlier block.

Key features
------------
- 9 edge function types: MLP, tree, piecewise, polynomial, periodic, RBF,
  log/exp, quadratic, product
- Multi-dimensional nodes (Pareto-distributed dim, partial observation)
- Concat-then-transform or per-parent + aggregate modes
- Power-law feature importances, Kumaraswamy warping, random rescaling
- Optional ExtraTrees learnability filter
"""

import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Edge functions
# ---------------------------------------------------------------------------

def _get_activations():
    return [
        lambda x: x,
        np.tanh,
        lambda x: 1 / (1 + np.exp(-np.clip(x, -20, 20))),
        lambda x: np.log1p(np.abs(x)),
        np.abs,
        np.sin,
        np.cos,
        lambda x: x ** 2,
        lambda x: x * (1 / (1 + np.exp(-1.702 * np.clip(x, -20, 20)))),
        lambda x: np.where(x > 0, x, 0.01 * x),
        lambda x: np.exp(-x ** 2),
        lambda x: np.log1p(np.exp(np.clip(x, -20, 20))),
        lambda x: np.maximum(x, 0),
        lambda x: x / (1 + np.exp(-np.clip(x, -20, 20))),
        lambda x: np.sign(x) * np.sqrt(np.abs(x)),
        lambda x: np.clip(x, -1, 1),
        lambda x: x ** 3,
    ]

_ACTIVATIONS = _get_activations()


def _make_mlp_fn(in_dim, rng):
    n_layers = rng.integers(1, 4)
    width    = int(rng.integers(8, 65))
    act      = _ACTIVATIONS[rng.integers(0, len(_ACTIVATIONS))]
    Ws, bs, dims = [], [], [in_dim]
    for i in range(n_layers):
        out = width if i < n_layers - 1 else 1
        lim = np.sqrt(6.0 / (dims[-1] + out))
        Ws.append(rng.uniform(-lim, lim, (dims[-1], out)))
        bs.append(rng.uniform(-0.1, 0.1, out))
        dims.append(out)
    def mlp_fn(x):
        h = x.reshape(-1, 1) if x.ndim == 1 else x.astype(float)
        for i, (W, b) in enumerate(zip(Ws, bs)):
            h = h @ W + b
            if i < n_layers - 1:
                h = act(h)
        return 50.0 * np.tanh(h.ravel() / 50.0)
    return mlp_fn


def _make_random_tree(rng, depth, max_depth):
    if depth >= max_depth or rng.random() < 0.25:
        return ('leaf', rng.uniform(-2, 2))
    return ('split', rng.uniform(-3, 3),
            _make_random_tree(rng, depth + 1, max_depth),
            _make_random_tree(rng, depth + 1, max_depth))

def _eval_tree(tree, x):
    if tree[0] == 'leaf':
        return np.full(x.shape, tree[1])
    _, thr, left, right = tree
    out  = np.empty_like(x)
    mask = x <= thr
    if mask.any():    out[mask]  = _eval_tree(left,  x[mask])
    if (~mask).any(): out[~mask] = _eval_tree(right, x[~mask])
    return out

def _make_tree_fn(rng):
    tree = _make_random_tree(rng, 0, int(rng.integers(2, 7)))
    def tree_fn(x): return _eval_tree(tree, x.ravel())
    return tree_fn


def _make_piecewise_fn(rng):
    n    = int(rng.integers(2, 6))
    bpts = np.sort(rng.uniform(-3, 3, n - 1))
    sl   = rng.uniform(-2, 2, n)
    ic   = rng.uniform(-1, 1, n)
    def piecewise_fn(x):
        v = x.ravel()
        idx = np.searchsorted(bpts, v)
        return 50.0 * np.tanh((sl[idx] * v + ic[idx]) / 50.0)
    return piecewise_fn


def _make_poly_fn(rng):
    coeffs = rng.standard_normal(int(rng.integers(2, 5)) + 1) * 0.5
    def poly_fn(x):
        v = np.clip(x.ravel(), -5, 5)
        out = coeffs[-1]
        for c in coeffs[-2::-1]:
            out = out * v + c
        return out
    return poly_fn


def _make_periodic_fn(rng):
    freq  = rng.uniform(0.5, 5.0)
    phase = rng.uniform(0, 2 * np.pi)
    amp   = rng.uniform(0.5, 2.0)
    def periodic_fn(x): return amp * np.sin(freq * x.ravel() + phase)
    return periodic_fn


def _make_rbf_fn(rng):
    center = rng.uniform(-2, 2)
    sigma  = rng.uniform(0.3, 2.0)
    amp    = rng.uniform(0.5, 3.0)
    def rbf_fn(x):
        v = x.ravel()
        return amp * np.exp(-((v - center) ** 2) / (2 * sigma ** 2))
    return rbf_fn


def _make_logexp_fn(rng):
    a = rng.uniform(0.5, 2.0)
    if rng.random() < 0.5:
        def fn(x): return a * np.log1p(np.abs(x.ravel())) * np.sign(x.ravel())
    else:
        b = rng.uniform(0.3, 1.5)
        def fn(x): return a * (np.exp(b * np.clip(x.ravel(), -5, 5)) - 1)
    return fn


def _make_quadratic_fn(rng, in_dim=1):
    A = rng.standard_normal((in_dim, in_dim)) * 0.3
    M = (A + A.T) / 2
    w = rng.standard_normal(in_dim) * 0.5
    b = rng.uniform(-0.5, 0.5)
    if in_dim == 1:
        def qfn(x):
            v = np.clip(x.ravel(), -5, 5)
            return 50.0 * np.tanh((M[0, 0] * v**2 + w[0] * v + b) / 50.0)
    else:
        def qfn(x):
            xc = np.clip(x, -5, 5)
            return 50.0 * np.tanh((np.sum((xc @ M) * xc, 1) + xc @ w + b) / 50.0)
    return qfn


def _make_product_fn(rng):
    simple = [_make_piecewise_fn, _make_poly_fn, _make_periodic_fn,
              _make_rbf_fn, _make_logexp_fn]
    fn1 = simple[int(rng.integers(0, len(simple)))](rng)
    fn2 = simple[int(rng.integers(0, len(simple)))](rng)
    def product_fn(x): return 50.0 * np.tanh(fn1(x) * fn2(x) / 50.0)
    return product_fn


def _make_multidim_proj_fn(in_dim, rng):
    proj = rng.standard_normal(in_dim)
    proj /= np.linalg.norm(proj) + 1e-8
    fn = _make_edge_fn(1, rng)
    def mfn(x): return fn((x @ proj).reshape(-1, 1))
    return mfn


def _make_concat_fn(in_dim, rng):
    return _make_mlp_fn(in_dim, rng) if rng.random() < 0.5 else _make_quadratic_fn(rng, in_dim)


def _make_edge_fn(in_dim, rng):
    c = int(rng.integers(0, 9))
    return [
        lambda: _make_mlp_fn(in_dim, rng),
        lambda: _make_tree_fn(rng),
        lambda: _make_piecewise_fn(rng),
        lambda: _make_poly_fn(rng),
        lambda: _make_periodic_fn(rng),
        lambda: _make_rbf_fn(rng),
        lambda: _make_logexp_fn(rng),
        lambda: _make_quadratic_fn(rng),
        lambda: _make_product_fn(rng),
    ][c]()


# ---------------------------------------------------------------------------
# Root distributions
# ---------------------------------------------------------------------------

def _sample_root(n, rng):
    c = rng.integers(0, 7)
    if c == 0:
        return rng.standard_normal(n)
    elif c == 1:
        return rng.uniform(-3, 3, n)
    elif c == 2:
        return rng.beta(rng.uniform(0.5, 5), rng.uniform(0.5, 5), n)
    elif c == 3:
        return rng.lognormal(0, 1, n)
    elif c == 4:
        return rng.standard_t(rng.uniform(2, 10), n)
    elif c == 5:
        return rng.uniform(0.5, 2.0) * (1 + rng.pareto(rng.uniform(1.5, 5.0), n))
    else:
        k    = int(rng.integers(2, 4))
        w    = rng.dirichlet(np.ones(k))
        mu   = rng.uniform(-3, 3, k)
        sd   = rng.uniform(0.3, 2.0, k)
        asgn = rng.choice(k, n, p=w)
        out  = np.empty(n)
        for i in range(k):
            m = asgn == i
            out[m] = rng.normal(mu[i], sd[i], m.sum())
        return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(parent_values, rng):
    if len(parent_values) == 1:
        return parent_values[0]
    s = np.column_stack(parent_values)
    c = rng.integers(0, 6)
    if c == 0:
        return s.mean(1)
    elif c == 1:
        return s @ rng.dirichlet(np.ones(len(parent_values)))
    elif c == 2:
        d  = len(parent_values)
        W1 = rng.uniform(-1, 1, (d, d * 2))
        b1 = rng.uniform(-0.1, 0.1, d * 2)
        W2 = rng.uniform(-1, 1, (d * 2, 1))
        return (np.tanh(s @ W1 + b1) @ W2).ravel()
    elif c == 3:
        r = np.clip(s[:, 0], -10, 10)
        for j in range(1, s.shape[1]):
            r = np.clip(r * np.clip(s[:, j], -10, 10), -10, 10)
        return r
    elif c == 4:
        return s.max(1)
    else:
        cl = np.clip(s, -20, 20)
        return np.log(np.sum(np.exp(cl), 1) + 1e-8)


# ---------------------------------------------------------------------------
# Robust standardization
# ---------------------------------------------------------------------------

def _robust_standardize(v):
    if v.ndim == 1:
        s   = np.sort(v)
        n   = len(s)
        med = (s[n // 2] + s[(n - 1) // 2]) * 0.5
        dev = np.sort(np.abs(s - med))
        mad = (dev[n // 2] + dev[(n - 1) // 2]) * 0.5 * 1.4826
        sc  = mad if mad > 1e-8 else max(float(np.std(v)), 1e-8)
        return (v - med) / sc
    else:
        s   = np.sort(v, 0)
        n   = v.shape[0]
        med = ((s[n // 2] + s[(n - 1) // 2]) * 0.5)[np.newaxis, :]
        dev = np.sort(np.abs(v - med), 0)
        mad = ((dev[n // 2] + dev[(n - 1) // 2]) * 0.5 * 1.4826)[np.newaxis, :]
        sc  = np.where(mad < 1e-8, np.maximum(np.std(v, 0, keepdims=True), 1e-8), mad)
        return (v - med) / sc


# ---------------------------------------------------------------------------
# Local Causal Structure
# ---------------------------------------------------------------------------

class LocalCausalStructure:
    """A small topologically-ordered DAG (3-8 nodes) with random edge functions."""

    def __init__(self, n_nodes, external_indices, rng, *,
                 multidim=True, max_parents=4, external_d=None):
        self.n_nodes    = n_nodes
        self.rng        = rng
        self.multidim   = multidim
        self.max_parents = max_parents
        if external_d is None:
            external_d = {}

        self.parents     = {}
        self.edge_fns    = {}
        self.concat_mode = {}
        self.d_nodes     = {}

        for i in range(n_nodes):
            d = 1 + int(np.clip(rng.pareto(1.5), 0, 3)) if multidim else 1
            self.d_nodes[i] = d

            cands = ([(True, j) for j in range(i)]
                     + [(False, j) for j in external_indices])
            if not cands:
                self.parents[i]     = []
                self.concat_mode[i] = False
                continue

            n_par  = min(rng.integers(1, max_parents), len(cands))
            chosen = rng.choice(len(cands), n_par, replace=False)
            pars   = [cands[c] for c in chosen]
            self.parents[i] = pars

            def _pdim(is_int, idx, _i=i):
                return self.d_nodes[idx] if is_int else external_d.get(idx, 1)

            use_concat = len(pars) >= 2 and rng.random() < 0.4
            self.concat_mode[i] = use_concat
            if use_concat:
                total = sum(_pdim(ii, jj) for ii, jj in pars)
                self.edge_fns[i] = [_make_concat_fn(total, rng)]
            else:
                fns = []
                for is_int, idx in pars:
                    pd = _pdim(is_int, idx)
                    fns.append(_make_multidim_proj_fn(pd, rng) if pd > 1
                                else _make_edge_fn(1, rng))
                self.edge_fns[i] = fns

    def _expand(self, v1d, d, n):
        scales = np.clip(self.rng.normal(1.0, 0.35, d), 0.3, 1.7)
        sig    = max(float(np.std(v1d)), 1e-6)
        noise  = self.rng.standard_normal((n, d)) * sig * self.rng.uniform(0.2, 0.7)
        out    = v1d.reshape(-1, 1) * scales + noise
        if d > 1 and self.rng.random() < 0.25:
            Q, _ = np.linalg.qr(self.rng.standard_normal((d, d)))
            out  = out @ Q
        return out

    def generate(self, n_samples, external_values):
        """Generate node values.

        Parameters
        ----------
        n_samples:
            Number of rows to generate.
        external_values:
            Dict mapping global_idx → array [n_samples] or [n_samples, d].

        Returns
        -------
        list of arrays, one per node.
        """
        values = []
        for i in range(self.n_nodes):
            d = self.d_nodes[i]
            if not self.parents[i]:
                if d == 1:
                    values.append(_robust_standardize(_sample_root(n_samples, self.rng)))
                else:
                    cols = [_sample_root(n_samples, self.rng) for _ in range(d)]
                    rv   = np.column_stack(cols)
                    if self.rng.random() < 0.5:
                        Q, _ = np.linalg.qr(self.rng.standard_normal((d, d)))
                        rv   = rv @ Q
                    values.append(_robust_standardize(rv))
                continue

            if self.concat_mode[i]:
                arrs = []
                for is_int, idx in self.parents[i]:
                    pv = values[idx] if is_int else external_values[idx]
                    arrs.append(pv.reshape(-1, 1) if pv.ndim == 1 else pv)
                cin    = np.clip(np.column_stack(arrs), -10, 10)
                result = _robust_standardize(self.edge_fns[i][0](cin).ravel())
            else:
                contribs = []
                for (is_int, idx), fn in zip(self.parents[i], self.edge_fns[i]):
                    pv = values[idx] if is_int else external_values[idx]
                    contribs.append(fn(pv.reshape(-1, 1) if pv.ndim == 1 else pv))
                result = _robust_standardize(_aggregate(contribs, self.rng))

            result = np.clip(np.where(np.isfinite(result), result, 0.0), -100, 100)
            values.append(self._expand(result, d, n_samples) if d > 1 else result)
        return values


# ---------------------------------------------------------------------------
# Hierarchical SCM builder
# ---------------------------------------------------------------------------

def build_hierarchical_scm(n_features, rng, *, multidim=True, max_parents=4):
    """Build a hierarchical SCM of multiple LCS blocks.

    Parameters
    ----------
    n_features:
        Number of features; the SCM will have n_features + 1 nodes total
        (last node = target).
    rng:
        numpy Generator.
    multidim:
        Whether nodes can be multi-dimensional.
    max_parents:
        Maximum parents per node.

    Returns
    -------
    lcs_list:
        List of (LocalCausalStructure, offset) tuples.
    global_parents:
        Dict {global_idx: [parent_global_idx, ...]}.
    global_d:
        Dict {global_idx: dimensionality}.
    """
    target_nodes = n_features + 1
    lcs_list, global_d, global_parents = [], {}, {}
    offset = 0

    while offset < target_nodes:
        remaining = target_nodes - offset
        n_nodes   = min(int(rng.integers(3, 9)), remaining)
        if remaining - n_nodes < 3 and remaining <= 8:
            n_nodes = remaining

        lcs = LocalCausalStructure(n_nodes, list(range(offset)), rng,
                                   multidim=multidim, max_parents=max_parents,
                                   external_d=global_d)
        lcs_list.append((lcs, offset))
        for j in range(n_nodes):
            gidx = offset + j
            global_d[gidx] = lcs.d_nodes[j]
            global_parents[gidx] = [
                (offset + idx if is_int else idx)
                for is_int, idx in lcs.parents[j]
            ]
        offset += n_nodes

    return lcs_list, global_parents, global_d


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(all_values, global_parents, n_features, rng):
    """Extract X and y_raw from DAG node values.

    Target = last node. Observation priority: direct parents of target →
    other ancestors → rest. Partial observation of multi-dim nodes creates
    natural Bayes error.

    Returns
    -------
    X : ndarray [n_samples, n_features]
    y_raw : ndarray [n_samples]
    """
    all_idx    = sorted(all_values.keys())
    target_idx = all_idx[-1]

    # BFS ancestors of target
    ancestors = set()
    frontier  = list(global_parents.get(target_idx, []))
    while frontier:
        node = frontier.pop()
        if node not in ancestors:
            ancestors.add(node)
            frontier.extend(global_parents.get(node, []))

    direct  = set(global_parents.get(target_idx, []))
    other   = ancestors - direct
    non_anc = [i for i in all_idx if i != target_idx and i not in ancestors]
    rng.shuffle(non_anc)
    order = sorted(direct) + sorted(other) + non_anc

    cols, n_so_far = [], 0
    n_samples = all_values[target_idx].shape[0]
    for idx in order:
        if n_so_far >= n_features:
            break
        v = all_values[idx]
        if v.ndim == 1:
            cols.append(v.reshape(-1, 1))
            n_so_far += 1
        else:
            d         = v.shape[1]
            remaining = n_features - n_so_far
            n_obs     = max(max(1, int(d * 0.75)), min(d, remaining))
            n_obs     = min(n_obs, remaining)
            chosen    = rng.choice(d, n_obs, replace=False) if n_obs < d else np.arange(d)
            cols.append(v[:, chosen])
            n_so_far += n_obs

    X = np.column_stack(cols) if cols else rng.standard_normal((n_samples, 1))
    if X.shape[1] > n_features:
        X = X[:, :n_features]
    elif X.shape[1] < n_features:
        X = np.column_stack([X, rng.standard_normal((n_samples, n_features - X.shape[1]))])

    tv = all_values[target_idx]
    if tv.ndim > 1:
        w     = rng.standard_normal(tv.shape[1])
        y_raw = tv @ (w / (np.linalg.norm(w) + 1e-8))
    else:
        y_raw = tv.ravel()

    return X.astype(np.float64), y_raw.astype(np.float64)


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------

def apply_power_law_importances(X, rng, *, mild=False):
    n = X.shape[1]
    if n <= 1:
        return X
    q     = np.exp(rng.uniform(np.log(0.3 if mild else 0.5),
                                np.log(2.0 if mild else 5.0)))
    sigma = np.exp(rng.uniform(np.log(1e-3), np.log(1.0 if mild else 2.0)))
    ranks = np.arange(1, n + 1, dtype=float)
    w     = np.abs(ranks ** (-q) * np.exp(rng.normal(0, sigma, n)))
    w     = np.maximum(w, 0.05 * w.max())
    w     = w / (w.sum() + 1e-8) * n
    rng.shuffle(w)
    return X * w[np.newaxis, :]


def apply_kumaraswamy(X, col, rng):
    a    = np.exp(rng.uniform(np.log(0.2), np.log(5.0)))
    b    = np.exp(rng.uniform(np.log(0.2), np.log(5.0)))
    v    = X[:, col]
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if vmax - vmin < 1e-8:
        return
    vs       = np.clip((v - vmin) / (vmax - vmin), 1e-8, 1 - 1e-8)
    X[:, col] = 1.0 - (1.0 - vs ** a) ** b


def postprocess(X, rng, *, task_type='cls'):
    n = X.shape[1]
    if n >= 2 and rng.random() < 0.85:
        X = apply_power_law_importances(X, rng, mild=(task_type == 'cls'))
    if n >= 2 and rng.random() < 0.4:
        n_warp = max(1, int(n * rng.uniform(0.1, 0.5)))
        for col in rng.choice(n, min(n_warp, n), replace=False):
            apply_kumaraswamy(X, col, rng)
    if n >= 2 and rng.random() < 0.5:
        X = X * np.exp(rng.uniform(np.log(0.1), np.log(10.0), n))[np.newaxis, :]
    return X


def discretize_some(X, rng):
    nf = X.shape[1]
    if nf < 2:
        return X
    n_disc = int(nf * rng.uniform(0.3, 0.9))
    if n_disc == 0:
        return X
    for col in rng.choice(nf, n_disc, replace=False):
        n_bins = int(rng.integers(2, 21))
        edges  = np.quantile(X[:, col], np.linspace(0, 1, n_bins + 1)[1:-1])
        X[:, col] = np.digitize(X[:, col], np.unique(edges)).astype(float)
    return X


# ---------------------------------------------------------------------------
# Reg2Cls
# ---------------------------------------------------------------------------

def reg2cls(y_raw, n_classes, rng):
    """Convert continuous target to integer class labels via quantile thresholds."""
    roll = rng.random()
    if roll < 0.35:
        percentiles = np.linspace(0, 100, n_classes + 1)
    elif roll < 0.75:
        alpha = rng.uniform(0.5, 3.0)
        props = rng.dirichlet(np.full(n_classes, alpha))
        percentiles = np.concatenate([[0.0], np.cumsum(props[:-1]) * 100, [100.0]])
    else:
        dom  = int(rng.integers(0, n_classes))
        dp   = rng.uniform(0.70, 0.92)
        props = np.full(n_classes, (1 - dp) / max(n_classes - 1, 1))
        props[dom] = dp
        percentiles = np.concatenate([[0.0], np.cumsum(props[:-1]) * 100, [100.0]])

    thresholds = np.percentile(y_raw, percentiles[1:-1])
    labels     = np.digitize(y_raw, thresholds).astype(np.float32)

    if rng.random() < 0.4:
        rate  = rng.uniform(0.01, 0.12)
        n_flip = max(1, int(len(labels) * rate))
        idx   = rng.choice(len(labels), n_flip, replace=False)
        labels[idx] = rng.integers(0, n_classes, n_flip).astype(np.float32)

    return labels


# ---------------------------------------------------------------------------
# Learnability filter
# ---------------------------------------------------------------------------

def is_learnable(X, y, task_type):
    """Return True if an ExtraTrees model can beat a constant baseline."""
    try:
        from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    except ImportError:
        return True
    if X.shape[0] < 20:
        return True
    Xc = np.nan_to_num(X.astype(np.float32), nan=0.0)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if task_type == 'cls':
                m = ExtraTreesClassifier(15, max_depth=6, bootstrap=True,
                                          oob_score=True, random_state=0, n_jobs=1)
                m.fit(Xc, y.astype(int))
                return m.oob_score_ > 1.0 / len(np.unique(y)) + 0.05
            else:
                m = ExtraTreesRegressor(15, max_depth=6, bootstrap=True,
                                         oob_score=True, random_state=0, n_jobs=1)
                m.fit(Xc, y.astype(float))
                return m.oob_score_ > 0.02
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Dataset and batch generators
# ---------------------------------------------------------------------------

def generate_dataset(n_samples, n_features, task_type, n_classes=None, *,
                     rng=None, multidim=True, apply_filter=True,
                     max_retries=3):
    """Generate one synthetic tabular dataset from the hierarchical LCS prior.

    Parameters
    ----------
    n_samples : int
    n_features : int
    task_type : 'cls' or 'reg'
    n_classes : int or None  (sampled randomly for 'cls' if None)
    rng : numpy Generator or None
    multidim : bool  — allow multi-dimensional SCM nodes
    apply_filter : bool  — reject unlearnable datasets (ExtraTrees OOB check)
    max_retries : int

    Returns
    -------
    dict with keys: X, y, task_type, n_classes, filtered
    """
    if rng is None:
        rng = np.random.default_rng()
    if task_type == 'cls' and n_classes is None:
        n_classes = int(rng.integers(2, 11))

    for attempt in range(max_retries + 1):
        try:
            lcs_list, global_parents, _ = build_hierarchical_scm(
                n_features, rng,
                multidim=multidim,
                max_parents=int(rng.integers(3, 6)),
            )
            all_values = {}
            for lcs, offset in lcs_list:
                node_vals = lcs.generate(n_samples, dict(all_values))
                for j, v in enumerate(node_vals):
                    v = np.clip(np.where(np.isfinite(v), v, 0.0), -100, 100)
                    all_values[offset + j] = v

            X, y_raw = extract_features(all_values, global_parents, n_features, rng)
        except Exception:
            continue

        X = postprocess(X, rng, task_type=task_type)
        if rng.random() < 0.6:
            X = discretize_some(X, rng)

        if task_type == 'cls':
            y            = reg2cls(y_raw, n_classes, rng)
            unique       = np.unique(y)
            n_cls_actual = len(unique)
            if n_cls_actual < n_classes:
                m = {o: n for n, o in enumerate(unique)}
                y = np.array([m[v] for v in y], dtype=np.float32)
        else:
            y   = y_raw.astype(np.float32)
            ys  = np.std(y)
            if ys > 1e-8:
                y = (y - np.mean(y)) / ys
            y            = np.clip(y, -1e4, 1e4)
            n_cls_actual = None

        # Winsorize X
        X = _winsorize(X.astype(np.float32))

        filtered = attempt > 0
        if apply_filter and not is_learnable(X, y, task_type):
            filtered = True
            continue

        return {'X': X, 'y': y, 'task_type': task_type,
                'n_classes': n_cls_actual if task_type == 'cls' else None,
                'filtered':  filtered}

    # Fallback
    X = np.random.default_rng().standard_normal((n_samples, n_features)).astype(np.float32)
    y = (rng.integers(0, n_classes, n_samples).astype(np.float32) if task_type == 'cls'
         else rng.standard_normal(n_samples).astype(np.float32))
    return {'X': X, 'y': y, 'task_type': task_type,
            'n_classes': n_classes if task_type == 'cls' else None,
            'filtered':  True}


def _winsorize(X):
    """Clip each column to ±6 MAD, then hard-clip to ±1e4."""
    for col in range(X.shape[1]):
        v   = X[:, col]
        fin = v[np.isfinite(v)]
        if len(fin) < 5:
            continue
        med = np.median(fin)
        mad = np.median(np.abs(fin - med)) * 1.4826
        if mad < 1e-8:
            mad = max(np.std(fin), 1e-8)
        mask = np.isfinite(v)
        X[mask, col] = np.clip(v[mask], med - 6 * mad, med + 6 * mad)
    return np.clip(np.where(np.isfinite(X), X, 0.0), -1e4, 1e4)


def generate_batch(batch_size, n_samples, n_features, task_type, *,
                   n_classes=None, rng=None, apply_filter=False):
    """Generate a batch of datasets.

    Returns
    -------
    X_batch : ndarray [batch_size, n_samples, n_features]
    y_batch : ndarray [batch_size, n_samples]
    n_classes : int or None
    """
    if rng is None:
        rng = np.random.default_rng()
    X_list, y_list, actual_n_classes = [], [], n_classes
    for _ in range(batch_size):
        d = generate_dataset(n_samples, n_features, task_type,
                              n_classes=n_classes, rng=rng,
                              apply_filter=apply_filter)
        X_list.append(d['X'])
        y_list.append(d['y'])
        if actual_n_classes is None and task_type == 'cls':
            actual_n_classes = d['n_classes']
    return np.stack(X_list), np.stack(y_list), actual_n_classes
