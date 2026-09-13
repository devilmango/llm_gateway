from __future__ import annotations

import pytest

from llm_gateway.cache import EmbeddingCache
from llm_gateway.evaluation import retrieval_metrics


def test_retrieval_metrics_compute_recall_mrr_and_ndcg():
    metrics = retrieval_metrics(["a", "b", "c", "d"], ["b", "d"], [1, 3, 4])

    assert metrics["recall@1"] == 0
    assert metrics["recall@3"] == 0.5
    assert metrics["recall@4"] == 1
    assert metrics["mrr"] == 0.5
    assert metrics["ndcg@1"] == 0
    assert metrics["ndcg@3"] == pytest.approx((1 / 1.584962500721156) / (1 + 1 / 1.584962500721156))
    assert metrics["ndcg@4"] == pytest.approx((1 / 1.584962500721156 + 1 / 2.321928094887362) /
                                               (1 + 1 / 1.584962500721156))


def test_retrieval_metrics_reject_invalid_labels():
    with pytest.raises(ValueError, match="unique"):
        retrieval_metrics(["a", "a"], ["a"], [1])
    with pytest.raises(ValueError, match="relevant"):
        retrieval_metrics(["a"], [], [1])


@pytest.mark.asyncio
async def test_embedding_cache_is_tenant_provider_model_and_pricing_scoped():
    cache = EmbeddingCache()
    first = cache.key("secret query", tenant="tenant-a", provider="openai", model="embed-v1",
                      pricing_version="v1", secret="stable-key")
    assert first != cache.key("secret query", tenant="tenant-b", provider="openai", model="embed-v1",
                              pricing_version="v1", secret="stable-key")
    assert first != cache.key("secret query", tenant="tenant-a", provider="openai", model="embed-v1",
                              pricing_version="v2", secret="stable-key")
    assert "secret query" not in first
    await cache.put(first, [0.1, 0.2], ttl_seconds=30)
    assert await cache.get(first) == [0.1, 0.2]
