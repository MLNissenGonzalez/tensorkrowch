"""
Tests for tt_rss_of_rows: TT-RSS of a tensor given on its support.

Ported from tpm-tn-benchmark (``tests/unit/core/test_initialisers.py`` at 4951c5b), plus the
rank options, ``svd``, ``generator`` and the rank rule shared with ``truncated_svd``.
"""

import itertools
import math
import time

import pytest

import torch
import tensorkrowch as tk
from tensorkrowch.decompositions import tt_rss_of_rows
from tensorkrowch.decompositions.tt_decompositions import TIE_TOLERANCE


# 300 rows of a chain of 6 variables with 3 values (107 distinct, unequal counts): samples of
# tpm-tn-benchmark's TreeLatentModel(chain(6), latent_dim=2, vocab_size=3, concentration=0.5,
# seed=0).sample(300, seed=1), written out so that the tests do not depend on that repository.
# √p̂ of them has ranks (3, 8, 17, 6, 3).
CHAIN_ROWS = """
    020220 212100 020220 121012 020212 221000 222100 120212 220212 220212 222100 201012
    220212 020010 112210 202100 012100 122000 020210 020012 020212 020220 120210 212210
    220210 020212 220212 220210 212100 020012 020212 022100 021000 120212 022100 112120
    212100 220212 120210 220210 212100 022100 211000 212100 112121 222100 020212 012100
    202100 012120 200220 121020 022020 020010 220212 222100 212101 012100 220000 200220
    222100 020212 121010 201000 220212 020210 020210 000220 212100 112100 220212 022000
    212100 222212 020212 212100 220220 200212 211000 220210 120210 020212 221000 211000
    212100 020221 200212 220212 000000 222100 022121 212100 220212 222212 220000 212120
    012100 122101 211000 221000 120000 220212 000212 222100 201212 220200 020021 222112
    212210 212100 120210 112100 200000 220212 200012 022100 212100 212100 220212 022101
    120100 012100 220212 122100 120221 220221 012012 222010 220000 220212 222100 000212
    220210 000212 222100 111020 221012 020012 201000 001012 222100 201012 221012 122100
    012110 212000 211212 122000 012000 220212 220212 222120 020000 201000 012100 220010
    211001 220212 020212 220212 112100 020210 002100 212210 022100 200212 220212 120100
    001001 000210 200210 011021 120000 121020 222200 022100 220000 211220 220210 202100
    220210 020212 120212 012210 120000 211020 122100 220220 000212 220220 222100 222100
    220210 220012 012100 220210 020212 120212 212100 220212 220210 220020 120210 220212
    222121 220100 022100 212100 222100 212121 212100 011000 020212 200021 220000 222112
    120210 212100 200012 212121 220012 020000 020212 112100 220212 220100 220220 212100
    022112 121021 201000 220220 220212 220020 212110 220221 020212 020000 000212 222112
    220212 220212 020212 211212 211012 112100 012200 002100 222120 111010 221012 212012
    022212 220212 211210 220212 111210 222101 120212 212000 212112 112100 212000 020112
    111012 122100 212100 020220 120212 012112 122112 020010 221210 212200 020212 220212
    200010 220212 220000 201212 020212 200221 000000 022100 120210 200212 220212 220012
    000212 220212 220100 112200 022121 120212 200221 022100 222100 121012 120021 022100
"""


def _chain_rows():
    return torch.tensor([[int(c) for c in row] for row in CHAIN_ROWS.split()])


def _skewed_rows():
    """300 uniform rows of 6 variables (3 values) whose x_0 is 1 on 6 of them and 0 elsewhere:
    under cum_percentage 0.9 the first cut keeps one component, and the second at most 3."""
    x = torch.randint(0, 3, (300, 6), generator=torch.Generator().manual_seed(0))
    x[:, 0] = 0
    x[:6, 0] = 1
    return x


def _distinct_rows():
    """80 distinct rows on a chain of 10 (3 values), each seen once: the cuts' singular values
    come in groups of equal ones."""
    x = torch.unique(torch.randint(0, 3, (80, 10), generator=torch.Generator().manual_seed(5)), dim=0)
    assert len(x) == 80
    return x


def _amplitudes(x):
    """The distinct rows of ``x`` and √p̂ on them."""
    rows, counts = torch.unique(x, dim=0, return_counts=True)
    return rows, (counts.double() / len(x)).sqrt()


def _dense(rows, amplitudes, dims):
    dense = torch.zeros(dims, dtype=torch.float64)
    dense[tuple(rows.T)] = amplitudes
    return dense


def _contracted(tensors):
    """The dense tensor of a chain's tensors, legs (left, value, right)."""
    dense = tensors[0]
    for tensor in tensors[1:]:
        dense = torch.tensordot(dense, tensor, dims=([dense.dim() - 1], [0]))
    return dense


def _ranks(tensors):
    return [t.shape[-1] for t in tensors[:-1]]


def _numerical_rank(values, size):
    return int((values > values.max() * size * torch.finfo(values.dtype).eps).sum())


def _unfolding(dense, k):
    return dense.reshape(math.prod(dense.shape[:k + 1]), -1)


def _numerical_ranks(dense):
    """The numerical rank of each unfolding of ``dense``, after variable k."""
    ranks = []
    for k in range(dense.dim() - 1):
        matrix = _unfolding(dense, k)
        ranks.append(_numerical_rank(torch.linalg.svdvals(matrix), max(matrix.shape)))
    return ranks


def _split_ties(dense, ranks):
    """The cuts (after variable k) where the rank r_k falls inside a group of equal singular
    values (within TIE_TOLERANCE of the largest) of the unfolding of ``dense`` there."""
    split = []
    for k, r in enumerate(ranks):
        matrix = _unfolding(dense, k)
        values = torch.linalg.svdvals(matrix)
        kept = _numerical_rank(values, max(matrix.shape))
        if r < kept and values[r - 1] - values[r] <= TIE_TOLERANCE * values[0]:
            split.append(k)
    return split


def _tt_rss_reference(x, d=3, **options):
    """``tt_rss`` on √p̂ of the rows ``x`` as a function (every row a sketch sample, a float64
    one-hot embedding at one d for every variable, domain arange(d)): its chain contracted to
    a dense tensor, and its edges' ranks."""
    def sqrt_phat(q):
        counts = (q.long()[:, None, :] == x[None, :, :]).all(dim=-1).sum(dim=1)
        return (counts.double() / len(x)).sqrt().unsqueeze(-1)

    def one_hot(q):
        return torch.nn.functional.one_hot(q.long(), num_classes=d).to(torch.float64)

    tensors = tk.decompositions.tt_rss(function=sqrt_phat, embedding=one_hot, sketch_samples=x,
                                       domain=torch.arange(d), dtype=torch.float64, verbose=False, **options)
    return _contracted(tensors), _ranks(tensors)


def _assert_tt_rss(x, rows, amplitudes, tensors, **options):
    """``tensors`` are ``tt_rss``'s on √p̂ of ``x`` (dense, both normalised), where no cut
    splits a tie: each edge's rank is ``tt_rss``'s, at most √p̂'s numerical rank across it
    (``tt_rss`` keeps singular vectors at rounding level there, which change nothing), and
    at most r_{k-1} d_k (``tt_rss``'s cap, implied)."""
    ranks = _ranks(tensors)
    dense = _dense(rows, amplitudes, [3] * 6)
    assert _split_ties(dense, ranks) == []
    reference, reference_ranks = _tt_rss_reference(x, **options)
    ours = _contracted(tensors)
    assert torch.allclose(ours / ours.norm(), reference / reference.norm(), rtol=1e-10, atol=1e-12)
    assert ranks == [min(a, b) for a, b in zip(reference_ranks, _numerical_ranks(dense))]
    assert all(r <= previous * 3 for previous, r in zip([1] + ranks, ranks))
    return ranks, ours, dense


# MARK: ported

@pytest.mark.parametrize("data, rank, cum_percentage", [
    ("chain", 1, None), ("chain", 2, None), ("chain", 3, None), ("chain", 5, None), ("chain", 8, None),
    ("chain", 17, None), ("chain", 27, None), ("chain", 5, 0.99), ("chain", 27, 0.99), ("chain", 27, 0.9),
    ("chain", 100, 0.9), ("skewed", 27, 0.9), ("skewed", 27, 0.95)])
def test_is_tt_rss_without_a_split_tie(data, rank, cum_percentage):
    """The chain rows (unequal counts), and rows whose first variable is nearly constant:
    where no cut falls inside a group of equal singular values, the chain is ``tt_rss``'s on
    √p̂, at ranks below √p̂'s ((3, 8, 17, 6, 3) on the chain rows) and at and above them, with
    and without cum_percentage; above √p̂'s ranks the chain is √p̂."""
    x = _chain_rows() if data == "chain" else _skewed_rows()
    rows, amplitudes = _amplitudes(x)
    tensors = tt_rss_of_rows(rows, amplitudes, [3] * 6, rank, cum_percentage=cum_percentage)
    ranks, ours, dense = _assert_tt_rss(x, rows, amplitudes, tensors, rank=rank, cum_percentage=cum_percentage)
    if rank >= 17 and cum_percentage is None:
        assert ranks == [3, 8, 17, 6, 3]
        assert torch.allclose(ours, dense, rtol=1e-10, atol=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("rank", [8, 16])
def test_with_a_split_tie_is_tt_rss_in_distribution(rank):
    """80 distinct rows on a chain of 10, every row seen once: at ranks 8 and 16 some cut falls
    inside a group of equal singular values, where ``tt_rss`` itself changes with the seed.
    Over 12 fixed seeds the mean of |⟨T, √p̂⟩| / ‖T‖ is within 3 standard errors (of the
    difference) of ``tt_rss``'s, and the same seed gives the same T."""
    x = _distinct_rows()
    rows, amplitudes = _amplitudes(x)
    dense = _dense(rows, amplitudes, [3] * 10)
    overlaps = {"ours": [], "tt_rss": []}
    for seed in range(12):
        torch.manual_seed(seed)
        tensors = tt_rss_of_rows(rows, amplitudes, [3] * 10, rank)
        if seed == 0:
            assert _split_ties(dense, _ranks(tensors)) != []
            torch.manual_seed(seed)
            assert torch.equal(_contracted(tensors), _contracted(tt_rss_of_rows(rows, amplitudes, [3] * 10, rank)))
        torch.manual_seed(seed)
        for name, t in (("ours", _contracted(tensors)), ("tt_rss", _tt_rss_reference(x, rank=rank)[0])):
            overlaps[name].append(((t * dense).sum().abs() / t.norm()).item())
    ours, reference = (torch.tensor(overlaps[name], dtype=torch.float64) for name in ("ours", "tt_rss"))
    error = (ours.var() / 12 + reference.var() / 12).sqrt()
    assert ours.std() > 0
    assert (ours.mean() - reference.mean()).abs() <= 3 * error, (ours.mean(), reference.mean(), error)


def test_keeps_no_singular_value_at_rounding_level():
    """Every configuration of 4 variables of 3 values once: the tensor is uniform, a product,
    so each cut has one block of rank 1, whose other singular values are at rounding level.
    At rank 8 every edge has rank 1 and the chain is the tensor."""
    rows = torch.tensor(list(itertools.product(range(3), repeat=4)))
    amplitudes = torch.full((len(rows),), 1 / 9, dtype=torch.float64)
    tensors = tt_rss_of_rows(rows, amplitudes, [3] * 4, 8)
    assert _ranks(tensors) == [1, 1, 1]
    assert torch.allclose(_contracted(tensors), torch.full([3] * 4, 1 / 9, dtype=torch.float64), rtol=1e-12)


def test_splits_a_group_of_values_equal_within_the_tolerance_at_random():
    """Amplitudes 1, 1 − 1e-14 and 1 − 2e-14 on a diagonal of 3 (two variables): one group at
    TIE_TOLERANCE, so at rank 1 the kept vector is a random unit vector of the three, and T
    changes with the seed; with distinct values (1, 0.5, 0.25) it does not."""
    rows = torch.tensor([[0, 0], [1, 1], [2, 2]])
    for amplitudes, moves in ((torch.tensor([1, 1 - 1e-14, 1 - 2e-14], dtype=torch.float64), True),
                              (torch.tensor([1, 0.5, 0.25], dtype=torch.float64), False)):
        dense = []
        for seed in (0, 1):
            torch.manual_seed(seed)
            dense.append(_contracted(tt_rss_of_rows(rows, amplitudes, [3, 3], 1)))
        assert (not torch.allclose(dense[0], dense[1], rtol=1e-6, atol=1e-6)) is moves


def test_keeps_a_share_of_every_singular_value():
    """cum_percentage is a share of all of the cut's squared singular values, not of those
    the rank keeps: squares 8, 4, 2, 1 on a diagonal of 4, rank 3, cum_percentage 0.85. Of
    all (15) two keep 0.8, so three; of the three the rank keeps (14) two would keep 0.86."""
    rows = torch.tensor([[v, v] for v in range(4)])
    amplitudes = torch.tensor([8.0, 4.0, 2.0, 1.0], dtype=torch.float64).sqrt()
    tensors = tt_rss_of_rows(rows, amplitudes, [4, 4], 3, cum_percentage=0.85)
    assert tensors[0].shape[-1] == 3


def test_keeps_the_signs_of_the_amplitudes():
    """Signed amplitudes, at full rank, T = A: the cut after the first variable has a block
    with one suffix and two prefixes (rows (0, 0) and (1, 0), amplitudes −1 and 2), whose
    vector carries the signs, and one with one entry (row (2, 1), −3)."""
    rows = torch.tensor([[0, 0], [1, 0], [2, 1]])
    amplitudes = torch.tensor([-1.0, 2.0, -3.0], dtype=torch.float64)
    dense = _dense(rows, amplitudes, [3, 2])
    assert torch.allclose(_contracted(tt_rss_of_rows(rows, amplitudes, [3, 2], 3)), dense, rtol=1e-12, atol=1e-15)


def test_of_one_variable_is_the_tensor():
    """One variable: one tensor (d_0,), A itself, repeated rows added."""
    rows = torch.tensor([[2], [0], [2]])
    tensors = tt_rss_of_rows(rows, torch.tensor([0.5, -1.0, 0.25], dtype=torch.float64), [4], 3)
    assert len(tensors) == 1
    assert torch.equal(tensors[0], torch.tensor([-1.0, 0.0, 0.75, 0.0], dtype=torch.float64))


def test_under_rounding_of_the_amplitudes_is_the_same_at_the_same_seed():
    """80 distinct rows on a chain of 10 at rank 16, where cuts split groups of equal values:
    amplitudes moved by a relative 1e-15 (rounding, another machine) reorder the values
    within a group, and the same seed still gives the same T (within 1e-12 of its largest
    entry): the draw goes to the group's vectors by their blocks' order, not by their
    values'."""
    rows, amplitudes = _amplitudes(_distinct_rows())
    moved = amplitudes * (1 + 1e-15 * torch.randn(len(amplitudes), generator=torch.Generator().manual_seed(1),
                                                  dtype=torch.float64))
    assert not torch.equal(moved, amplitudes)
    dense = []
    for a in (amplitudes, moved):
        torch.manual_seed(0)
        dense.append(_contracted(tt_rss_of_rows(rows, a, [3] * 10, 16)))
    assert (dense[0] - dense[1]).abs().max() <= 1e-12 * dense[0].abs().max()


def test_forms_no_dense_sketch():
    """300 distinct rows on a chain of 43 with 23 values (SMILES's shape) at rank 8, in under
    10 s: ``tt_rss``'s sketches would be (distinct prefixes) × d × (distinct suffixes) per
    cut, every one evaluated."""
    x = torch.unique(torch.randint(0, 23, (300, 43), generator=torch.Generator().manual_seed(0)), dim=0)
    rows, amplitudes = _amplitudes(x)
    start = time.perf_counter()
    tensors = tt_rss_of_rows(rows, amplitudes, [23] * 43, 8)
    assert time.perf_counter() - start < 10
    assert _ranks(tensors) == [8] * 42


@pytest.mark.parametrize("change, match", [
    (lambda rows, a: dict(rows=rows.double()), "integers"), (lambda rows, a: dict(dims=[3] * 5), "integers"),
    (lambda rows, a: dict(amplitudes=a[:-1]), "one entry per row"),
    (lambda rows, a: dict(amplitudes=a.to(torch.complex128)), "real"),
    (lambda rows, a: dict(amplitudes=torch.zeros_like(a)), "zero"), (lambda rows, a: dict(dims=[2] * 6), "outside"),
    (lambda rows, a: dict(rank=0), "rank"), (lambda rows, a: dict(rank=1.5), "rank"),
    (lambda rows, a: dict(rank=[4] * 4), "n - 1"), (lambda rows, a: dict(rank=[4, 4, 0, 4, 4]), "rank"),
    (lambda rows, a: dict(rank=[4, None, 4, 4, 4]), "rank"),
    (lambda rows, a: dict(cutoff=-1.0), "cutoff"), (lambda rows, a: dict(atol=-0.1), "atol"),
    (lambda rows, a: dict(rtol=1.1), "rtol"), (lambda rows, a: dict(rtol=-0.1), "rtol"),
    (lambda rows, a: dict(cum_percentage=1.5), "cum_percentage"),
    (lambda rows, a: dict(cum_percentage=-0.1), "cum_percentage")])
def test_refuses_what_it_cannot_take(change, match):
    """Rows that are not (M, n) integers within ``dims``, amplitudes that are not a real
    vector of one entry per row or are zero on every row; a rank below 1 or not an int, a
    rank sequence of the wrong length; tolerances outside the ranges ``truncated_svd``
    refuses."""
    rows, amplitudes = _amplitudes(_chain_rows())
    call = {**dict(rows=rows, amplitudes=amplitudes, dims=[3] * 6, rank=4), **change(rows, amplitudes)}
    with pytest.raises(ValueError, match=match):
        tt_rss_of_rows(**call)


def test_adds_repeated_rows():
    """A tensor given by its support: a row given twice holds the sum of its amplitudes."""
    rows, amplitudes = _amplitudes(_chain_rows())
    twice = tt_rss_of_rows(torch.cat([rows, rows[:5]]), torch.cat([amplitudes, amplitudes[:5]]), [3] * 6, 27)
    added = amplitudes.clone()
    added[:5] *= 2
    once = tt_rss_of_rows(rows, added, [3] * 6, 27)
    assert torch.allclose(_contracted(twice), _contracted(once), rtol=1e-10, atol=1e-12)


# MARK: gaps of tpm-tn-benchmark#125's review

def test_a_group_with_one_value_past_the_rank_is_split_at_random():
    rows = torch.tensor([[0, 0], [1, 1], [2, 2]])
    amplitudes = torch.ones(3, dtype=torch.float64)
    dense = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        dense.append(_contracted(tt_rss_of_rows(rows, amplitudes, [3, 3], 2)))
    assert not torch.allclose(dense[0], dense[1], atol=1e-6)


def test_a_value_at_rounding_level_never_joins_a_group():
    rows = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]])
    amplitudes = torch.tensor([1.0, 1.0, 1.0, 1.0, 1e-11], dtype=torch.float64)
    dense = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        dense.append(_contracted(tt_rss_of_rows(rows, amplitudes, [3, 3], 2)))
    assert torch.allclose(dense[0], dense[1], rtol=0, atol=1e-14)
    expected = torch.zeros(3, 3, dtype=torch.float64)
    expected[tuple(rows.T)] = amplitudes
    assert torch.allclose(dense[0], expected, rtol=0, atol=1e-14)


# MARK: rank options

@pytest.mark.parametrize("options", [
    dict(cutoff=0.05), dict(cutoff=0.1), dict(atol=1e-3), dict(atol=0.01), dict(atol=0.05),
    dict(rtol=0.01), dict(rtol=0.05), dict(rtol=0.1), dict(cum_percentage=0.9), dict(cum_percentage=0.99),
    dict(cutoff=0.05, atol=0.05), dict(rtol=0.01, cum_percentage=0.9)])
def test_each_tolerance_is_tt_rss_s(options):
    """``cutoff``, ``atol``, ``rtol`` and ``cum_percentage`` (alone and together, without a
    rank) give ``tt_rss``'s chain with the same option (``tt_rss`` needs a rank: 100, above
    every cut's), where no cut splits a tie; each binds at some cut (below (3, 8, 17, 6, 3))."""
    x = _chain_rows()
    rows, amplitudes = _amplitudes(x)
    tensors = tt_rss_of_rows(rows, amplitudes, [3] * 6, **options)
    ranks, _, _ = _assert_tt_rss(x, rows, amplitudes, tensors, rank=100, **options)
    assert ranks != [3, 8, 17, 6, 3]


def test_a_rank_sequence_caps_each_bond_by_its_own_entry():
    """Caps (2, 20, 5, 100, 1) on the chain rows (ranks (3, 8, 17, 6, 3)): each bond keeps
    the smaller of its own cap and its rank. Cut k's factor depends on its cap alone, so the
    first tensor (U_0) is the int rank 2's, and the last two cuts' tensors those of the
    sequence that differs only before them."""
    rows, amplitudes = _amplitudes(_chain_rows())
    tensors = tt_rss_of_rows(rows, amplitudes, [3] * 6, [2, 20, 5, 100, 1])
    assert _ranks(tensors) == [2, 8, 5, 6, 1]
    assert torch.equal(tensors[0], tt_rss_of_rows(rows, amplitudes, [3] * 6, 2)[0])
    other = tt_rss_of_rows(rows, amplitudes, [3] * 6, [3, 4, 9, 100, 1])
    assert all(torch.equal(a, b) for a, b in zip(tensors[4:], other[4:]))
    assert not torch.equal(tensors[2], other[2])


def test_a_sequence_equal_to_an_int_everywhere_is_the_int():
    rows, amplitudes = _amplitudes(_distinct_rows())
    for rank in (3, 8, 16):
        torch.manual_seed(0)
        as_int = tt_rss_of_rows(rows, amplitudes, [3] * 10, rank)
        torch.manual_seed(0)
        as_list = tt_rss_of_rows(rows, amplitudes, [3] * 10, [rank] * 9)
        torch.manual_seed(0)
        as_tuple = tt_rss_of_rows(rows, amplitudes, [3] * 10, (rank,) * 9)
        for other in (as_list, as_tuple):
            assert all(torch.equal(a, b) for a, b in zip(as_int, other))


@pytest.mark.parametrize("data", ["chain", "distinct", "signed"])
def test_without_a_criterion_the_chain_is_the_tensor(data):
    """``rank=None`` and no tolerance: every cut keeps its numerical rank, and the chain is
    the tensor to 1e-12, on distinct rows (every group of ties kept whole) too."""
    if data == "signed":
        x = torch.randint(0, 3, (200, 7), generator=torch.Generator().manual_seed(2))
        rows = torch.unique(x, dim=0)
        amplitudes = torch.randn(len(rows), generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    else:
        rows, amplitudes = _amplitudes(_chain_rows() if data == "chain" else _distinct_rows())
    dims = [3] * rows.shape[1]
    tensors = tt_rss_of_rows(rows, amplitudes, dims)
    dense = _dense(rows, amplitudes, dims)
    assert _ranks(tensors) == _numerical_ranks(dense)
    assert torch.allclose(_contracted(tensors), dense, rtol=0, atol=1e-12)


@pytest.mark.parametrize("options", [
    dict(), dict(rank=2), dict(rank=5), dict(cutoff=0.3), dict(atol=0.5), dict(rtol=0.2),
    dict(cum_percentage=0.7), dict(rank=4, cutoff=0.1, atol=0.05, rtol=0.01, cum_percentage=0.95)])
def test_the_rank_of_a_cut_is_truncated_svds(options):
    """The rank rule is ``truncated_svd``'s, on the cut's singular values sorted over all its
    blocks: on one cut (two variables; blocks 3 × 3, 2 × 2 and 1 × 1 of random amplitudes)
    the rank is the one ``truncated_svd`` keeps of the dense unfolding."""
    g = torch.Generator().manual_seed(4)
    blocks = [(range(0, 3), range(0, 3)), (range(3, 5), range(3, 5)), (range(5, 6), range(5, 6))]
    rows = torch.tensor([[i, j] for left, right in blocks for i in left for j in right])
    amplitudes = torch.randn(len(rows), generator=g, dtype=torch.float64) * torch.tensor(
        [1.0] * 9 + [0.7] * 4 + [0.4], dtype=torch.float64)
    dense = _dense(rows, amplitudes, [6, 6])
    expected = tk.utils.truncated_svd(dense, **options)[1].shape[-1]
    expected = min(expected, _numerical_rank(torch.linalg.svdvals(dense), 6))
    assert tt_rss_of_rows(rows, amplitudes, [6, 6], **options)[0].shape[-1] == expected


@pytest.mark.parametrize("options", [dict(rtol=0.01), dict(rtol=0.1), dict(cum_percentage=0.9),
                                     dict(cum_percentage=0.99), dict(rtol=0.05, cum_percentage=0.9)])
def test_the_rank_of_rtol_does_not_depend_on_the_scale(options):
    """``rtol`` and ``cum_percentage`` keep the same ranks of A × 2⁻⁶⁰, A and A × 2⁶⁰
    (the chain rows), and the same tensors up to that factor."""
    rows, amplitudes = _amplitudes(_chain_rows())
    chains = [tt_rss_of_rows(rows, amplitudes * c, [3] * 6, **options) for c in (2.0 ** -60, 1.0, 2.0 ** 60)]
    assert _ranks(chains[0]) == _ranks(chains[1]) == _ranks(chains[2]) != [3, 8, 17, 6, 3]
    for c, chain in zip((2.0 ** -60, 2.0 ** 60), (chains[0], chains[2])):
        assert torch.allclose(_contracted(chain) / c, _contracted(chains[1]), rtol=1e-10, atol=1e-14)


def test_cum_percentage_keeps_its_ranks_at_amplitudes_below_machine_epsilon():
    """300 uniform rows of a chain of 8 (3 values) at cum_percentage 0.9: the ranks at
    amplitudes × 1e-20 are those at × 1, not the numerical ranks (which an absolute clamp
    of the singular values at eps gave)."""
    x = torch.randint(0, 3, (300, 8), generator=torch.Generator().manual_seed(0))
    rows, amplitudes = _amplitudes(x)
    ranks = [_ranks(tt_rss_of_rows(rows, amplitudes * c, [3] * 8, cum_percentage=0.9)) for c in (1.0, 1e-20)]
    numerical = _ranks(tt_rss_of_rows(rows, amplitudes, [3] * 8))
    assert ranks[0] == ranks[1] == [3, 8, 22, 38, 22, 8, 3]
    assert numerical != ranks[0]


# MARK: svd

def _blocks_not_rank_one(rows, n):
    """Per cut, the connected components of the unfolding's entries (a row joins its prefix
    and its suffix) with at least two prefixes and two suffixes."""
    count = 0
    for k in range(n - 1):
        parent = {}

        def find(i):
            while parent.setdefault(i, i) != i:
                i = parent[i]
            return i

        for row in rows.tolist():
            a, b = find(("p", tuple(row[:k + 1]))), find(("s", tuple(row[k + 1:])))
            if a != b:
                parent[a] = b
        sides = {}
        for row in rows.tolist():
            root = find(("p", tuple(row[:k + 1])))
            prefixes, suffixes = sides.setdefault(root, (set(), set()))
            prefixes.add(tuple(row[:k + 1]))
            suffixes.add(tuple(row[k + 1:]))
        count += sum(len(p) > 1 and len(s) > 1 for p, s in sides.values())
    return count


@pytest.mark.parametrize("data, rank", [("chain", 5), ("chain", 27), ("distinct", 16)])
def test_svd_is_called_once_per_block_that_is_not_rank_one(monkeypatch, data, rank):
    """A counting ``svd`` is the only SVD: called once per block with at least two prefixes
    and two suffixes, with ``full_matrices=False``, and ``torch.linalg.svd`` never (it
    raises), split ties (``distinct``) included; the tensors are the default's."""
    rows, amplitudes = _amplitudes(_chain_rows() if data == "chain" else _distinct_rows())
    dims = [3] * rows.shape[1]
    torch.manual_seed(0)
    default = tt_rss_of_rows(rows, amplitudes, dims, rank)
    original, calls = torch.linalg.svd, []

    def counting(matrix, full_matrices=True):
        calls.append((tuple(matrix.shape), full_matrices))
        return original(matrix, full_matrices=full_matrices)

    def refuse(*args, **kwargs):
        raise AssertionError("torch.linalg.svd called")

    monkeypatch.setattr(torch.linalg, "svd", refuse)
    torch.manual_seed(0)
    tensors = tt_rss_of_rows(rows, amplitudes, dims, rank, svd=counting)
    assert len(calls) == _blocks_not_rank_one(rows, len(dims)) > 0
    assert all(shape[0] > 1 and shape[1] > 1 and full is False for shape, full in calls)
    assert all(torch.equal(a, b) for a, b in zip(default, tensors))


def test_an_svd_that_raises_makes_the_call_raise():
    rows, amplitudes = _amplitudes(_chain_rows())

    def failing(matrix, full_matrices=False):
        raise torch.linalg.LinAlgError("did not converge")

    with pytest.raises(torch.linalg.LinAlgError, match="did not converge"):
        tt_rss_of_rows(rows, amplitudes, [3] * 6, 5, svd=failing)


# MARK: generator

def test_two_generators_with_the_same_seed_give_the_same_tensors():
    """At a split tie (80 distinct rows, rank 8): the same seed, the same tensors; another
    seed, other ones."""
    rows, amplitudes = _amplitudes(_distinct_rows())
    a, b, c = (tt_rss_of_rows(rows, amplitudes, [3] * 10, 8, generator=torch.Generator().manual_seed(seed))
               for seed in (7, 7, 8))
    assert all(torch.equal(s, t) for s, t in zip(a, b))
    assert not torch.allclose(_contracted(a), _contracted(c), atol=1e-6)


def test_the_default_generator_is_torch_s():
    """``generator=None`` after ``torch.manual_seed(s)`` gives a generator seeded with s's
    tensors, and draws from torch's default generator (it moves); a generator given leaves
    the default one alone."""
    rows, amplitudes = _amplitudes(_distinct_rows())
    torch.manual_seed(11)
    default = tt_rss_of_rows(rows, amplitudes, [3] * 10, 8)
    after_default = torch.rand(1)
    given = tt_rss_of_rows(rows, amplitudes, [3] * 10, 8, generator=torch.Generator().manual_seed(11))
    assert all(torch.equal(s, t) for s, t in zip(default, given))
    torch.manual_seed(11)
    assert not torch.equal(torch.rand(1), after_default)
    torch.manual_seed(11)
    tt_rss_of_rows(rows, amplitudes, [3] * 10, 8, generator=torch.Generator().manual_seed(11))
    assert torch.equal(torch.rand(1), torch.rand(1, generator=torch.Generator().manual_seed(11)))


def test_a_split_tie_draws_the_q_of_a_gaussian_qr_with_the_signs_of_r():
    """Three equal amplitudes on a diagonal of 3, rank 2: three rank-one blocks, one group,
    nothing above it, so U_0 (the first tensor) is the draw itself, over the blocks in their
    order: Q of the QR of a (3, 2) Gaussian matrix from the generator, its columns
    multiplied by the signs of R's diagonal. A Haar isometry, as tpm-tn-benchmark draws it."""
    rows = torch.tensor([[0, 0], [1, 1], [2, 2]])
    tensors = tt_rss_of_rows(rows, torch.ones(3, dtype=torch.float64), [3, 3], 2,
                             generator=torch.Generator().manual_seed(3))
    q, r = torch.linalg.qr(torch.randn(3, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(3)))
    assert torch.equal(tensors[0], q * torch.sign(r.diagonal()))
