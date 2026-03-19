# struct-sdfm

Hierarchical TabICL — a simplified [TabICL](https://github.com/soda-inria/tabicl) clone that uses a **hierarchical Local Causal Structure (LCS)** synthetic data prior instead of the original flat MLP/Tree SCM.

## File structure

```
struct-sdfm/
├── pyproject.toml              Python package metadata and dependencies
├── train.py                    CLI training script
└── src/
    └── struct_sdfm/            Installable package (import as struct_sdfm)
        ├── __init__.py         Public API re-exports
        ├── prior.py            Hierarchical SCM data generator
        ├── model.py            TabICLModel architecture
        ├── train.py            Training loop and checkpoint utilities
        ├── predictor.py        Sklearn-compatible classifier and regressor
        └── _cli.py             Entry point for the struct-sdfm-train command
```

### `prior.py` — Synthetic data prior

Generates tabular datasets via a hierarchy of Local Causal Structure (LCS) blocks.
Each block is a small DAG (3–8 nodes) with random edge functions; later blocks can
depend on nodes from any earlier block.

Key components:

| Symbol | Description |
|---|---|
| `LocalCausalStructure` | Single LCS DAG block with multi-dim nodes |
| `build_hierarchical_scm` | Composes multiple LCS blocks into one SCM |
| `extract_features` | Extracts X and y from node values (causal-priority ordering) |
| `generate_dataset` | End-to-end single dataset generation |
| `generate_batch` | Batched dataset generation for training |

Edge function types: MLP, decision tree, piecewise-linear, polynomial, periodic
(sin), RBF, log/exp, quadratic (xᵀMx), product (f·g).

Postprocessing: power-law feature importances, Kumaraswamy warping, random
per-column rescaling, quantile discretisation, Reg2Cls conversion.

### `model.py` — Model architecture

Two-stage transformer:

1. **RowEncoder** — attends across features for each sample; outputs a single CLS
   vector per sample.
2. **ICLearner** — attends across samples with a causal mask that prevents context
   samples from seeing query labels and prevents query samples from attending to
   each other.

| Class / function | Description |
|---|---|
| `TabICLModel` | Full model (FeatureEmbedding → RowEncoder → ICLearner → head) |
| `create_model(**kwargs)` | Factory with sensible defaults |

Default config: `embed_dim=128`, `nhead=8`, `num_row_blocks=3`,
`num_icl_blocks=8`, `max_classes=10`.

### `train.py` — Training

Single-stage training on the hierarchical SCM prior — no curriculum or
staged-sequence-length scheduling.

| Symbol | Description |
|---|---|
| `train(model, ...)` | Main training loop (AdamW + cosine warmup) |
| `load_checkpoint(path)` | Load model and metadata from a `.pt` file |

### `predictor.py` — Sklearn interface

| Class | Description |
|---|---|
| `HierarchicalTabICLClassifier` | `fit / predict / predict_proba / score` |
| `HierarchicalTabICLRegressor` | Quantile-binned regression wrapper |

Both classes ensemble over random feature permutations of the context set.

## Installation

```bash
pip install -e .                   # core only (torch + numpy)
pip install -e ".[filter]"         # + scikit-learn (learnability filter)
pip install -e ".[dev]"            # + scikit-learn + pytest
```

## Usage

### Training

```bash
python train.py --steps 100000 --device cuda --save model.pt
# or after installation:
struct-sdfm-train --steps 100000 --device cuda --save model.pt
```

Key options:

```
--embed-dim    128     Model dimension
--row-blocks   3       RowEncoder depth
--icl-blocks   8       ICLearner depth
--max-classes  10      Output head width
--batch-size   32      Datasets per step
--lr           1e-4    Peak learning rate
--warmup       2000    Warmup steps
--filter               Enable ExtraTrees learnability filter (slower)
```

### Inference

```python
from struct_sdfm import HierarchicalTabICLClassifier

clf = HierarchicalTabICLClassifier(checkpoint='model.pt', n_estimators=4)
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)   # [n_test, n_classes]
preds = clf.predict(X_test)
```

### Data generation only

```python
from struct_sdfm import generate_dataset, generate_batch

dataset = generate_dataset(n_samples=256, n_features=20, task_type='cls')
X, y = dataset['X'], dataset['y']

X_batch, y_batch, n_classes = generate_batch(
    batch_size=32, n_samples=256, n_features=20, task_type='cls'
)
```
