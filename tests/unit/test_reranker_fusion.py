"""``fuse_ranks`` — pure rank-fusion math extracted from ``CrossEncoderReranker``
(ADR-0008 amendment, 2026-08-04). No model loaded here (NFR-L3-2).
"""

from __future__ import annotations

from bankassist.rag.stages.reranker import fuse_ranks


def test_agreement_wins_when_both_signals_rank_something_first():
    # index 0 is both RRF-rank 1 (first in input order) and CE-rank 1 (highest score).
    scores = [10.0, 1.0, 0.5]
    fused = fuse_ranks(scores)
    assert fused[0][1] == 0


def test_a_strong_rrf_rank_survives_a_middling_ce_score():
    # index 0: RRF-rank 1, but the CE model actively dislikes it (lowest score).
    # index 2: RRF-rank 3, CE loves it (highest score).
    # Neither signal alone should completely veto the other's top pick.
    scores = [-9.0, 0.0, 9.0]
    fused = fuse_ranks(scores)
    top_indices = [index for _, index, _ in fused[:2]]
    assert 0 in top_indices  # strong RRF rank still makes the cut
    assert 2 in top_indices  # strong CE score still makes the cut


def test_output_is_sorted_descending_by_fusion_score():
    scores = [3.0, -1.0, 5.0, 0.0]
    fused = fuse_ranks(scores)
    fusion_scores = [item[0] for item in fused]
    assert fusion_scores == sorted(fusion_scores, reverse=True)


def test_preserves_ce_score_in_output():
    scores = [2.5, -3.1]
    fused = fuse_ranks(scores)
    by_index = {index: ce_score for _, index, ce_score in fused}
    assert by_index[0] == 2.5
    assert by_index[1] == -3.1


def test_single_candidate():
    fused = fuse_ranks([1.0])
    assert len(fused) == 1
    assert fused[0][1] == 0


def test_regression_case_reranker_alone_would_have_dropped_the_right_answer():
    """Mirrors the real KYC OVD-list failure this fix addresses: the correct
    chunk (index 9) had the strongest RRF rank (position 1) among the
    candidates but the *worst* cross-encoder score. Fusion must not let the
    cross-encoder's single low score veto retrieval's strong consensus."""
    scores = [-8.9] + [float(i) for i in range(8, -1, -1)]  # index 0 scores lowest
    fused = fuse_ranks(scores)
    top_5_indices = [index for _, index, _ in fused[:5]]
    assert 0 in top_5_indices
