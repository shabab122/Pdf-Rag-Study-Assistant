from src.retrieval import RetrievedChunk, filter_relevant_chunks


def chunk(distance: float) -> RetrievedChunk:
    return RetrievedChunk("text", "notes.pdf", 1, distance)


def test_threshold_keeps_lower_distances_and_boundary() -> None:
    result = filter_relevant_chunks((chunk(0.2), chunk(0.75), chunk(0.9)), 0.75)
    assert [item.distance for item in result] == [0.2, 0.75]


def test_threshold_can_reject_every_candidate() -> None:
    result = filter_relevant_chunks((chunk(0.8), chunk(1.1)), 0.75)
    assert result == ()
