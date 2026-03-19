"""Hierarchical TabICL model architecture.

Two-stage transformer:
  1. RowEncoder  — attends across features for each sample, emits a CLS vector
  2. ICLearner   — attends across samples for in-context learning

The ICL attention mask ensures:
  - Context samples cannot attend to query samples (prevents label leakage)
  - Query samples cannot attend to each other (independent at inference time)
  - Query samples can freely attend to all context samples
"""

import torch
import torch.nn as nn


__all__ = ['TabICLModel', 'create_model']


class FeatureEmbedding(nn.Module):
    """Embed each (feature_position, value) pair to ``embed_dim`` dimensions.

    Continuous values are projected with a learned linear layer.
    NaN positions are replaced by a learned NaN token.
    Feature positions are added via a learned embedding table.
    """

    def __init__(self, max_features: int, embed_dim: int):
        super().__init__()
        self.max_features = max_features
        self.value_proj   = nn.Linear(1, embed_dim)
        self.nan_emb      = nn.Parameter(torch.zeros(embed_dim))
        self.feat_pos     = nn.Embedding(max_features, embed_dim)
        nn.init.normal_(self.feat_pos.weight, std=0.02)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X : Tensor [B, N, F]  — float, NaN for missing values

        Returns
        -------
        Tensor [B*N, f, D]  where f = min(F, max_features)
        """
        B, N, F = X.shape
        f        = min(F, self.max_features)
        X        = X[:, :, :f]
        nan_mask = torch.isnan(X)
        X_clean  = X.masked_fill(nan_mask, 0.0)

        val_emb  = self.value_proj(X_clean.reshape(B * N, f, 1))       # [B*N, f, D]
        nan_flag = nan_mask.reshape(B * N, f, 1).float()
        val_emb  = val_emb * (1.0 - nan_flag) + self.nan_emb * nan_flag

        pos_idx = torch.arange(f, device=X.device)
        pos_emb = self.feat_pos(pos_idx).unsqueeze(0).expand(B * N, -1, -1)
        return val_emb + pos_emb                                         # [B*N, f, D]


class RowEncoder(nn.Module):
    """Attend across features per sample; return a single CLS vector per sample."""

    def __init__(self, embed_dim: int, nhead: int, num_blocks: int,
                 ff_factor: int = 2, dropout: float = 0.0):
        super().__init__()
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        layer    = nn.TransformerEncoderLayer(
            embed_dim, nhead, embed_dim * ff_factor, dropout,
            batch_first=True, norm_first=True, activation='gelu',
        )
        self.enc = nn.TransformerEncoder(layer, num_blocks,
                                          enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [M, f, D]  where M = B*N

        Returns
        -------
        Tensor [M, D]
        """
        M      = x.shape[0]
        tokens = torch.cat([self.cls.expand(M, -1, -1), x], dim=1)
        return self.enc(tokens)[:, 0]


class LabelEmbedding(nn.Module):
    """Map class labels to embedding vectors.

    Index ``max_classes`` is reserved for the unknown/query token (-1 labels).
    """

    def __init__(self, max_classes: int, embed_dim: int):
        super().__init__()
        self.max_classes = max_classes
        self.emb         = nn.Embedding(max_classes + 1, embed_dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        labels : LongTensor [B, N]  — -1 for query samples

        Returns
        -------
        Tensor [B, N, D]
        """
        idx = torch.where(labels == -1,
                          torch.full_like(labels, self.max_classes),
                          labels.clamp(0, self.max_classes))
        return self.emb(idx)


class ICLearner(nn.Module):
    """Transformer over the sample sequence for in-context learning.

    Attention mask
    --------------
    - Context → query : **blocked** (prevents label leakage to context)
    - Query → query   : **blocked** (queries are independent at inference)
    - Context → context : allowed
    - Query → context   : allowed
    """

    def __init__(self, embed_dim: int, nhead: int, num_blocks: int,
                 ff_factor: int = 2, dropout: float = 0.0):
        super().__init__()
        layer    = nn.TransformerEncoderLayer(
            embed_dim, nhead, embed_dim * ff_factor, dropout,
            batch_first=True, norm_first=True, activation='gelu',
        )
        self.enc = nn.TransformerEncoder(layer, num_blocks,
                                          enable_nested_tensor=False)

    @staticmethod
    def _build_mask(n_ctx: int, n_qry: int, device) -> torch.Tensor:
        n    = n_ctx + n_qry
        mask = torch.zeros(n, n, dtype=torch.bool, device=device)
        mask[:n_ctx, n_ctx:] = True
        mask[n_ctx:, n_ctx:] = True
        return mask

    def forward(self, seq: torch.Tensor, n_ctx: int) -> torch.Tensor:
        """
        Parameters
        ----------
        seq : Tensor [B, n_ctx + n_qry, D]
        n_ctx : int

        Returns
        -------
        Tensor [B, n_ctx + n_qry, D]
        """
        n_qry = seq.shape[1] - n_ctx
        mask  = self._build_mask(n_ctx, n_qry, seq.device)
        return self.enc(seq, mask=mask)


class TabICLModel(nn.Module):
    """Hierarchical TabICL model.

    Parameters
    ----------
    max_features : int
        Maximum number of input feature columns (excess columns are trimmed).
    embed_dim : int
        Transformer model dimension.
    nhead : int
        Number of attention heads (must divide ``embed_dim``).
    num_row_blocks : int
        Depth of the per-row feature encoder.
    num_icl_blocks : int
        Depth of the in-context-learning transformer.
    ff_factor : int
        Feedforward expansion ratio.
    dropout : float
        Dropout probability applied in attention and feedforward layers.
    max_classes : int
        Number of output classes (output head width).
    """

    def __init__(
        self,
        max_features:   int   = 100,
        embed_dim:      int   = 128,
        nhead:          int   = 8,
        num_row_blocks: int   = 3,
        num_icl_blocks: int   = 8,
        ff_factor:      int   = 2,
        dropout:        float = 0.0,
        max_classes:    int   = 10,
    ):
        super().__init__()
        self.max_features = max_features
        self.max_classes  = max_classes
        self.embed_dim    = embed_dim

        self.feat_emb  = FeatureEmbedding(max_features, embed_dim)
        self.row_enc   = RowEncoder(embed_dim, nhead, num_row_blocks, ff_factor, dropout)
        self.label_emb = LabelEmbedding(max_classes, embed_dim)
        self.icl       = ICLearner(embed_dim, nhead, num_icl_blocks, ff_factor, dropout)
        self.out_norm  = nn.LayerNorm(embed_dim)
        self.out_head  = nn.Linear(embed_dim, max_classes)

    # ------------------------------------------------------------------
    def _encode(self, X: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Encode a set of (feature, label) pairs → per-sample representations.

        Parameters
        ----------
        X : [B, N, F]
        labels : [B, N] long  — -1 for query

        Returns
        -------
        [B, N, D]
        """
        B, N, _ = X.shape
        feat    = self.feat_emb(X)                       # [B*N, f, D]
        row     = self.row_enc(feat).reshape(B, N, -1)   # [B, N, D]
        return row + self.label_emb(labels)

    # ------------------------------------------------------------------
    @staticmethod
    def _align_features(X_ctx: torch.Tensor, X_qry: torch.Tensor,
                         max_features: int):
        """Pad or trim both tensors to the same feature dimension."""
        F = min(max(X_ctx.shape[2], X_qry.shape[2]), max_features)
        result = []
        for X in (X_ctx, X_qry):
            f = X.shape[2]
            if f < F:
                pad = torch.full((*X.shape[:2], F - f), float('nan'),
                                 dtype=X.dtype, device=X.device)
                X   = torch.cat([X, pad], 2)
            result.append(X[:, :, :F])
        return result[0], result[1]

    # ------------------------------------------------------------------
    def forward(
        self,
        X_ctx:   torch.Tensor,
        y_ctx:   torch.Tensor,
        X_query: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        X_ctx   : FloatTensor [B, n_ctx, F]
        y_ctx   : LongTensor  [B, n_ctx]   — class labels (0-indexed)
        X_query : FloatTensor [B, n_qry, F]

        Returns
        -------
        logits : FloatTensor [B, n_qry, max_classes]
        """
        B, n_ctx = X_ctx.shape[:2]
        n_qry    = X_query.shape[1]

        X_ctx, X_query = self._align_features(X_ctx, X_query, self.max_features)

        y_ctx   = y_ctx.long().clamp(0, self.max_classes - 1)
        y_query = torch.full((B, n_qry), -1, dtype=torch.long, device=X_query.device)

        ctx_repr = self._encode(X_ctx,   y_ctx)    # [B, n_ctx, D]
        qry_repr = self._encode(X_query, y_query)  # [B, n_qry, D]

        seq     = torch.cat([ctx_repr, qry_repr], 1)    # [B, n_ctx+n_qry, D]
        seq     = self.icl(seq, n_ctx)

        query_out = seq[:, n_ctx:]                       # [B, n_qry, D]
        return self.out_head(self.out_norm(query_out))   # [B, n_qry, max_classes]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model(**kwargs) -> TabICLModel:
    """Create a TabICLModel with keyword overrides over the default config.

    Example
    -------
    >>> model = create_model(embed_dim=256, num_icl_blocks=12)
    """
    defaults = dict(
        max_features=100,
        embed_dim=128,
        nhead=8,
        num_row_blocks=3,
        num_icl_blocks=8,
        ff_factor=2,
        dropout=0.0,
        max_classes=10,
    )
    defaults.update(kwargs)
    return TabICLModel(**defaults)
