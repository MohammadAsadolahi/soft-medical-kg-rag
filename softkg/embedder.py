"""Embedding backend with an on-disk cache.

Jina embeddings v3 is the default because of its *task adapters*: the same weights produce different
vectors depending on whether a string is being stored or being asked about
(``retrieval.passage`` vs ``retrieval.query``). That asymmetry matters more here than for ordinary
dense retrieval. Graph-side strings are terse noun phrases ("dietary cadmium intake"); query-side
strings are pattern slots and verb phrases ("FOOD lowers", "reduces risk of"). Embedding both with
one undifferentiated encoder puts them in the same distribution and blurs the distinction the graph
is built on.

Two operational details this class exists to enforce:

**Task asymmetry is not optional.** Graph text must be embedded as ``passage`` and query text as
``query``. Getting this backwards silently degrades every score and produces no error, so the two
paths are separate methods rather than a flag a caller can forget.

**Embeddings are cached and fingerprinted.** A graph build over a corpus of this size is five
subspaces times tens of thousands of strings, which is CPU-hours. The cache makes a rebuild free.
The fingerprint (model + task + max length) is stored beside the vectors, so a cache written with
different settings is rejected loudly rather than silently mixed with fresh vectors -- a stale cache
that merely *looks* valid is the most expensive failure mode available here.

All vectors are L2-normalised on write, so cosine similarity is a plain dot product everywhere
downstream and no scoring path has to renormalise.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "jinaai/jina-embeddings-v3"

# Truncation limits per role. Graph nodes are short noun phrases and verb phrases; padding them to
# 512 tokens wastes most of the compute in a build. Documents get the long budget because the dense
# baseline in ``evaluation/`` embeds whole abstracts.
#
# 32 covers every graph-side subspace without truncation, including the composite ones. Measured over
# the extracted graph, the longest string in the widest subspace ("head verb tail") is 20 words and
# the 99th percentile is 12 -- a direct consequence of the extraction prompt capping entities at 5-6
# words and verbs at 4. Raising this budget would only add padding.
MAX_LEN_NODE = 32
MAX_LEN_QUERY = 64
MAX_LEN_DOC = 512


class Embedder:
    """Lazy-loading Jina v3 wrapper with a persistent text -> vector cache.

    The model is loaded on first use so that importing the package, or running a build whose cache
    is already complete, does not pay the several-second model load.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        cache_dir: str | Path | None = None,
        device: str | None = None,
        batch_size: int = 32,
        threads: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.threads = threads
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None
        self._cache: dict[str, dict[str, np.ndarray]] = {}
        self._loaded_namespaces: set[str] = set()

    # -- model -------------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            import torch
            from transformers import AutoModel

            logger.info("loading %s (first use)", self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            self._model.eval()
            if self.device:
                self._model = self._model.to(self.device)
            # Leave a couple of cores free: a build often runs alongside other work, and saturating
            # every core made throughput *worse* in practice by starving the data loader.
            import os
            torch.set_num_threads(self.threads or max(1, (os.cpu_count() or 4) - 2))
        return self._model

    @property
    def dim(self) -> int:
        return 1024  # jina-embeddings-v3

    # -- cache -------------------------------------------------------------
    def _fingerprint(self, task: str, max_len: int) -> str:
        return hashlib.sha1(
            f"{self.model_name}|{task}|{max_len}".encode()).hexdigest()[:12]

    def _namespace(self, task: str, max_len: int) -> str:
        return f"{task.replace('.', '_')}_{max_len}"

    def _load_namespace(self, ns: str, task: str, max_len: int) -> dict[str, np.ndarray]:
        """Load one cache namespace from disk, rejecting a fingerprint mismatch."""
        if ns in self._loaded_namespaces:
            return self._cache.setdefault(ns, {})
        self._loaded_namespaces.add(ns)
        store = self._cache.setdefault(ns, {})
        if not self.cache_dir:
            return store

        base = self.cache_dir / ns
        keys_path, vecs_path, meta_path = (base / "keys.json", base / "vectors.npy",
                                           base / "meta.json")
        if not (keys_path.exists() and vecs_path.exists() and meta_path.exists()):
            return store

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = self._fingerprint(task, max_len)
        if meta.get("fingerprint") != expected:
            raise RuntimeError(
                f"embedding cache at {base} was written with a different configuration "
                f"(cache={meta.get('fingerprint')}, expected={expected}). "
                f"Its vectors are not comparable to fresh ones -- delete the directory to rebuild.")

        keys = json.loads(keys_path.read_text(encoding="utf-8"))
        vectors = np.load(vecs_path)
        if len(keys) != vectors.shape[0]:
            raise RuntimeError(
                f"embedding cache at {base} is inconsistent: {len(keys)} keys vs "
                f"{vectors.shape[0]} vectors -- delete the directory to rebuild.")
        store.update(zip(keys, vectors))
        logger.info("embedding cache %s: loaded %d vectors", ns, len(keys))
        return store

    def _persist_namespace(self, ns: str, task: str, max_len: int) -> None:
        if not self.cache_dir:
            return
        store = self._cache.get(ns) or {}
        if not store:
            return
        base = self.cache_dir / ns
        base.mkdir(parents=True, exist_ok=True)
        keys = list(store)
        matrix = np.stack([store[k] for k in keys]).astype(np.float32)
        # Write to a temporary name and replace, so an interrupted save cannot leave a cache whose
        # keys and vectors disagree. The temp name must itself end in .npy -- np.save appends the
        # extension when it is missing, and would otherwise write to a path we do not then rename.
        tmp = base / "vectors.tmp.npy"
        np.save(tmp, matrix)
        tmp.replace(base / "vectors.npy")
        (base / "keys.json").write_text(json.dumps(keys, ensure_ascii=False), encoding="utf-8")
        (base / "meta.json").write_text(json.dumps({
            "fingerprint": self._fingerprint(task, max_len),
            "model": self.model_name, "task": task, "max_len": max_len, "count": len(keys),
        }, indent=1), encoding="utf-8")
        logger.info("embedding cache %s: saved %d vectors", ns, len(keys))

    # -- encoding ----------------------------------------------------------
    def _encode(self, texts: Sequence[str], task: str, max_len: int,
                *, persist: bool = True) -> np.ndarray:
        ns = self._namespace(task, max_len)
        store = self._load_namespace(ns, task, max_len)

        # Deduplicate before encoding. Entity strings repeat heavily across a corpus -- the same
        # ~20k distinct entities span ~27k triplets and appear in several subspaces -- so this
        # removes real work, not just bookkeeping.
        missing = sorted({t for t in texts if t and t not in store})
        if missing:
            logger.info("encoding %d new strings (task=%s, max_len=%d)",
                        len(missing), task, max_len)
            vectors = np.asarray(
                self.model.encode(missing, task=task, max_length=max_len,
                                  batch_size=self.batch_size),
                dtype=np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            # Jina returns normalised vectors already; renormalise defensively so downstream dot
            # products are cosine regardless of backend behaviour, and guard the zero-vector case.
            vectors = vectors / np.maximum(norms, 1e-12)
            store.update(zip(missing, vectors))
            if persist:
                self._persist_namespace(ns, task, max_len)

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            if t:
                out[i] = store[t]
        return out

    # -- public API: the two roles are separate on purpose ------------------
    def encode_nodes(self, texts: Sequence[str]) -> np.ndarray:
        """Embed graph-side strings (entities, verbs, entity pairs) as passages."""
        return self._encode(list(texts), "retrieval.passage", MAX_LEN_NODE)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed full documents as passages. Used by the dense baseline, not by the graph."""
        return self._encode(list(texts), "retrieval.passage", MAX_LEN_DOC)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Embed query-side strings (user questions, pattern slots) as queries."""
        return self._encode(list(texts), "retrieval.query", MAX_LEN_QUERY)

    def encode_query(self, text: str) -> np.ndarray:
        """Single query string, returned as a 1-D vector."""
        return self.encode_queries([text])[0]
