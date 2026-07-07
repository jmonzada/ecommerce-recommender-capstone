"""Stage-1 candidate generators (capstone Step 4).

All models fit on an InteractionMatrix built from an explicit interaction set,
so the two artifact builds the evaluation plan requires stay separate:
leave-last-order-out artifacts for the candidate-generator comparison, and
feature-window-only artifacts for ranker features and the end-to-end holdout
evaluation.

Scoring is batched: nothing here ever materialises a dense users x items
matrix (~25 GB for Olist) - per-user score vectors only.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, normalize

SEED = 42


class InteractionMatrix:
    """Binary customer x product matrix with stable id/index mappings.

    Built from deduped (customer_unique_id, product_id) pairs; the product axis
    covers the FULL catalogue (passed explicitly) so item indices are shared
    across models fitted on different interaction sets.
    """

    def __init__(self, pairs: pd.DataFrame, product_ids: np.ndarray):
        self.product_ids = np.asarray(product_ids)
        self.item_index = pd.Index(self.product_ids)
        self.user_ids = pairs["customer_unique_id"].unique()
        self.user_index = pd.Index(self.user_ids)
        rows = self.user_index.get_indexer(pairs["customer_unique_id"])
        cols = self.item_index.get_indexer(pairs["product_id"])
        keep = cols >= 0
        data = np.ones(keep.sum(), dtype=np.float32)
        self.X = sparse.csr_matrix(
            (data, (rows[keep], cols[keep])),
            shape=(len(self.user_ids), len(self.product_ids)),
        )
        self.X.data[:] = 1.0  # binary even if duplicate pairs slip through

    def user_items(self, customer_id) -> np.ndarray:
        i = self.user_index.get_indexer([customer_id])[0]
        if i < 0:
            return np.array([], dtype=int)
        return self.X[i].indices


class PopularityRecommender:
    """Training-set purchase counts; one ranking shared by every user."""

    def fit(self, im: InteractionMatrix):
        self.scores_ = np.asarray(im.X.sum(axis=0)).ravel()
        return self

    def scores(self, im: InteractionMatrix, user_items: np.ndarray) -> np.ndarray:
        return self.scores_


class ItemItemCF:
    """Cosine similarity on the co-purchase matrix; user score = sum of
    similarities to the user's training items."""

    def fit(self, im: InteractionMatrix):
        xn = normalize(im.X, axis=0)  # column-normalise -> item-item cosine
        s = (xn.T @ xn).tocsr()
        s.setdiag(0.0)
        s.eliminate_zeros()
        self.S = s
        return self

    def scores(self, im: InteractionMatrix, user_items: np.ndarray) -> np.ndarray:
        if len(user_items) == 0:
            return np.zeros(self.S.shape[0], dtype=np.float32)
        return np.asarray(self.S[user_items].sum(axis=0)).ravel()

    def pair_scores(self, user_items_map: dict, pairs: pd.DataFrame, item_index) -> np.ndarray:
        """Co-purchase similarity for arbitrary (customer, product) pairs -
        the ranker's CF-signal feature."""
        out = np.zeros(len(pairs), dtype=np.float32)
        cols = item_index.get_indexer(pairs["product_id"])
        s_csc = self.S.tocsc()
        for i, (cust, col) in enumerate(zip(pairs["customer_unique_id"], cols)):
            items = user_items_map.get(cust)
            if items is None or len(items) == 0 or col < 0:
                continue
            out[i] = s_csc[items, col].sum()
        return out


class ContentBased:
    """Cosine similarity in product CONTENT space: category one-hot plus the
    standardised static attributes. Deliberately excludes interaction-derived
    features (price, review means) so the model is window-independent and
    reaches products with zero sales history - that cold-start reach is its
    role in the lineup."""

    NUMERIC = [
        "product_photos_qty", "product_name_lenght", "product_description_lenght",
        "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
    ]

    def fit(self, im: InteractionMatrix, prod_features: pd.DataFrame):
        pf = prod_features.reindex(im.item_index)
        num = StandardScaler().fit_transform(pf[self.NUMERIC].fillna(0.0))
        cats = pd.get_dummies(pf["category"], dtype=np.float32).to_numpy()
        self.P = normalize(np.hstack([cats, num]).astype(np.float32))
        return self

    def scores(self, im: InteractionMatrix, user_items: np.ndarray) -> np.ndarray:
        if len(user_items) == 0:
            return np.zeros(self.P.shape[0], dtype=np.float32)
        profile = self.P[user_items].mean(axis=0)
        return self.P @ profile


class SVDRecommender:
    """Truncated SVD on the binary interaction matrix. Expected to hover near
    the popularity baseline at ~1.1 interactions/user - kept as a comparative
    experiment, per the evaluation plan."""

    def __init__(self, n_components: int = 64):
        self.n_components = n_components

    def fit(self, im: InteractionMatrix):
        svd = TruncatedSVD(n_components=self.n_components, random_state=SEED)
        self.U = svd.fit_transform(im.X)
        self.V = svd.components_.T.astype(np.float32)
        self.user_index = im.user_index
        return self

    def scores(self, im: InteractionMatrix, user_items: np.ndarray, customer_id=None) -> np.ndarray:
        i = self.user_index.get_indexer([customer_id])[0] if customer_id is not None else -1
        if i < 0:
            return np.zeros(self.V.shape[0], dtype=np.float32)
        return self.V @ self.U[i].astype(np.float32)


class HybridRecommender:
    """Per-user z-normalised blend: w * item-item CF + (1 - w) * content."""

    def __init__(self, cf: ItemItemCF, content: ContentBased, w: float = 0.5):
        self.cf, self.content, self.w = cf, content, w

    @staticmethod
    def _z(s: np.ndarray) -> np.ndarray:
        sd = s.std()
        return (s - s.mean()) / sd if sd > 0 else s

    def scores(self, im: InteractionMatrix, user_items: np.ndarray) -> np.ndarray:
        return self.w * self._z(self.cf.scores(im, user_items)) + (1 - self.w) * self._z(
            self.content.scores(im, user_items)
        )


def top_k_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, ordered best-first (argpartition, no full sort)."""
    k = min(k, len(scores))
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]
