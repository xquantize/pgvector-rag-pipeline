"""Unit tests for retrieval (mock embedding + db layers)."""

from unittest.mock import patch

from rag.retrieval import Chunk, retrieve


def test_retrieve_orders_by_ascending_distance_and_respects_k():
    fake_rows = [
        {"content": "a", "source": "s1.txt", "metadata": {}, "distance": 0.1},
        {"content": "b", "source": "s2.txt", "metadata": {"k": 1}, "distance": 0.2},
        {"content": "c", "source": "s3.txt", "metadata": {}, "distance": 0.3},
    ]

    with (
        patch("rag.retrieval.embed_query", return_value=[0.0] * 8) as embed,
        patch("rag.retrieval.db.search", return_value=fake_rows[:2]) as search,
    ):
        results = retrieve("what is a?", k=2)

    embed.assert_called_once_with("what is a?")
    search.assert_called_once()
    assert search.call_args.args[1] == 2
    assert len(results) == 2
    assert all(isinstance(r, Chunk) for r in results)
    assert [r.distance for r in results] == [0.1, 0.2]
    assert results[0].source == "s1.txt"
    assert results[1].metadata == {"k": 1}
