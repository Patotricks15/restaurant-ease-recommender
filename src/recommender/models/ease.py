"""EASE^R: Embarrassingly Shallow Autoencoder for Sparse Recommendation.

EASE^R (Steck, 2019) learns a single item-item weight matrix ``B`` in
**closed form** — no gradient descent, no iterative optimisation — by
solving a ridge-regularised linear regression that reconstructs the binary
interaction matrix ``X`` from itself:

    min_B || X - X B ||_F^2 + lambda * ||B||_F^2,  s.t. diag(B) = 0.

The zero-diagonal constraint prevents the trivial solution ``B = I``. The
closed-form solution is

    P = (X^T X + lambda * I)^-1
    B = P / (-diag(P))       (elementwise, column-broadcast)
    B[diag] = 0

and a user's scores over every item are simply ``X[u] @ B``: the predicted
affinity for an item is a weighted sum of the user's other positive items.
This is linear algebra on a dense ``n_items x n_items`` matrix, not a neural
network, and needs no backpropagation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from recommender.data.splitting import Split


@dataclass
class EASERecommender:
    """Closed-form item-item similarity model.

    Attributes:
        l2_reg: Ridge regularisation strength (``lambda`` above). Larger
            values shrink off-diagonal weights more aggressively.
    """

    l2_reg: float = 300.0

    B: np.ndarray | None = None
    popularity: np.ndarray | None = None
    n_users: int = 0
    n_items: int = 0

    def fit(self, split: Split) -> EASERecommender:
        """Solve for ``B`` in closed form on the training matrix.

        Args:
            split: Leakage-safe partition supplying the training matrix.

        Returns:
            ``self``, fitted.

        Raises:
            ValueError: If ``l2_reg`` is not positive.
        """
        if self.l2_reg <= 0:
            raise ValueError(f"l2_reg must be positive, got {self.l2_reg}.")

        users, items = split.train_matrix_entries()
        x = np.zeros((split.n_users, split.n_items), dtype=np.float64)
        x[users, items] = 1.0

        gram = x.T @ x
        diag_indices = np.diag_indices(gram.shape[0])
        gram[diag_indices] += self.l2_reg
        p = np.linalg.inv(gram)

        b = p / (-np.diag(p))
        b[diag_indices] = 0.0

        self.B = b
        self.popularity = x.sum(axis=0)
        self.n_users = split.n_users
        self.n_items = split.n_items
        return self

    def _check_fitted(self) -> None:
        if self.B is None or self.popularity is None:
            raise RuntimeError("Model is not fitted; call fit() first.")

    def score_user_from_history(self, history: set[int] | np.ndarray) -> np.ndarray:
        """Score every restaurant from an arbitrary set of positive items.

        Because EASE^R scores a user purely from their interaction vector
        rather than from a learned per-user embedding, this also handles a
        cold user with history unseen at training time: pass their known
        positives directly instead of a training-time user index.
        """
        self._check_fitted()
        assert self.B is not None
        x_u = np.zeros(self.n_items, dtype=np.float64)
        if len(history):
            x_u[list(history)] = 1.0
        return x_u @ self.B

    def recommend(
        self, user_index: int, k: int = 10, exclude: set[int] | None = None  # noqa: ARG002
    ) -> np.ndarray:
        """Return top-scoring unseen restaurant indices for a known user's history.

        ``user_index`` is accepted for interface parity with the other
        recommenders in this portfolio, but EASE^R scores purely from the
        interaction vector: the caller's ``exclude`` set *is* that history,
        which doubles as both the scoring input and the items to suppress.
        """
        if k < 1:
            raise ValueError("k must be positive.")
        history = exclude or set()
        scores = self.score_user_from_history(history).copy()
        if history:
            scores[list(history)] = -np.inf
        k = min(k, int(np.isfinite(scores).sum()))
        if k == 0:
            return np.array([], dtype=np.int64)
        top = np.argpartition(-scores, k - 1)[:k]
        return top[np.argsort(-scores[top], kind="stable")]
