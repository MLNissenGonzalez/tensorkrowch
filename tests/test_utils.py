"""
This script contains tests for utils:

    * TestTruncatedSVD
"""

import pytest

import torch
import tensorkrowch as tk


class TestTruncatedSVD:  # MARK: TestTruncatedSVD

    @pytest.fixture
    def diag_tensor(self):
        # Singular values are exactly [5.0, 3.0, 1.0, 0.1]
        return torch.diag(torch.tensor([5.0, 3.0, 1.0, 0.1]))

    @pytest.mark.parametrize(
        'kwargs, expected_rank',
        [
            ({}, 4),
            ({'rank': 2}, 2),
            ({'cutoff': 1.0}, 2),
            ({'atol': 1.05}, 2),
            ({'rtol': 0.03}, 2),
            ({'cum_percentage': 0.97}, 2),
            ({'rank': 3, 'cutoff': 1.0, 'atol': 1.05}, 2),
            ({'rank': 4, 'rtol': 0.03, 'cum_percentage': 0.97}, 2),
        ],
    )
    def test_truncated_svd_rank_selection(self, diag_tensor, kwargs, expected_rank):
        u, s, vh = tk.utils.truncated_svd(diag_tensor, **kwargs)

        assert u.shape == (4, expected_rank)
        assert s.shape == (expected_rank,)
        assert vh.shape == (expected_rank, 4)
        assert torch.equal(s, torch.tensor([5.0, 3.0, 1.0, 0.1])[:expected_rank])

    def test_truncated_svd_batched_rank_selection(self):
        tensor = torch.stack([
            torch.diag(torch.tensor([5.0, 3.0, 1.0, 0.1])),
            torch.diag(torch.tensor([4.0, 2.0, 0.5, 0.05])),
        ])

        u, s, vh = tk.utils.truncated_svd(tensor, cutoff=0.5, atol=1.0)

        # cutoff gives rank 3 (because one batch has 1.0), atol gives rank 3 -> final rank 3
        assert u.shape == (2, 4, 3)
        assert s.shape == (2, 3)
        assert vh.shape == (2, 3, 4)

    @pytest.mark.parametrize(
        'kwargs',
        [
            {'rank': 0},
            {'rank': 1.5},
            {'cutoff': -1.0},
            {'cutoff': '1'},
            {'atol': -0.1},
            {'rtol': -0.1},
            {'rtol': 1.1},
            {'cum_percentage': -0.1},
            {'cum_percentage': 1.1},
        ],
    )
    def test_truncated_svd_invalid_arguments(self, diag_tensor, kwargs):
        with pytest.raises(ValueError):
            tk.utils.truncated_svd(diag_tensor, **kwargs)

    @pytest.mark.parametrize('dtype, scales', [
        (torch.float64, (2.0 ** -60, 1.0, 2.0 ** 60)),
        (torch.float32, (2.0 ** -60, 1.0, 2.0 ** 60)),
    ])
    @pytest.mark.parametrize('kwargs', [
        {'rtol': 0.0}, {'rtol': 0.01}, {'rtol': 0.2}, {'cum_percentage': 0.5},
        {'cum_percentage': 0.95}, {'cum_percentage': 1.0},
        {'rtol': 0.05, 'cum_percentage': 0.9},
    ])
    def test_truncated_svd_rtol_rank_does_not_depend_on_the_scale(self, dtype, scales, kwargs):
        """The rank ``rtol`` and ``cum_percentage`` keep of s, c s is the same: a
        matrix of rank 2 (the other singular values at rounding level) and one
        with graded singular values, at powers of two (exact), where squaring
        unscaled values would underflow (float32's graded tail at 2⁻⁶⁰)."""
        g = torch.Generator().manual_seed(0)
        low_rank = (torch.randn(8, 2, generator=g, dtype=dtype)
                    @ torch.randn(2, 8, generator=g, dtype=dtype))
        graded = torch.diag(torch.tensor([1.0, 0.5, 0.2, 0.1, 1e-3, 1e-4, 1e-5, 1e-6],
                                         dtype=dtype))
        for matrix in (low_rank, graded):
            ranks = [tk.utils.truncated_svd(matrix * c, **kwargs)[1].shape[-1]
                     for c in scales]
            assert ranks[0] == ranks[1] == ranks[2], ranks

    def test_truncated_svd_rtol_below_machine_epsilon_keeps_the_unit_scale_rank(self):
        """float32, largest singular value 1e-12 (below eps): the rank kept is
        the one of the matrix at scale 1, not every value (as an absolute clamp
        at eps gave)."""
        g = torch.Generator().manual_seed(0)
        matrix = torch.randn(8, 2, generator=g) @ torch.randn(2, 8, generator=g)
        for kwargs in ({'rtol': 0.01}, {'cum_percentage': 0.95}):
            assert tk.utils.truncated_svd(matrix * 1e-12, **kwargs)[1].shape[-1] == \
                tk.utils.truncated_svd(matrix, **kwargs)[1].shape[-1] == 2

    @pytest.mark.parametrize('kwargs, expected_rank', [
        ({'rtol': 0.2}, 4), ({'rtol': 0.5}, 2), ({'rtol': 0.8}, 1),
        ({'cum_percentage': 0.9}, 4), ({'cum_percentage': 0.4}, 2),
    ])
    def test_truncated_svd_rtol_of_a_zero_matrix(self, kwargs, expected_rank):
        """An all-zero matrix: every value clamped at eps, as before, so the tail
        of k of 4 values has share k / 4."""
        assert tk.utils.truncated_svd(torch.zeros(4, 4), **kwargs)[1].shape[-1] == expected_rank

    @staticmethod
    def _rtol_matrices(dtype):
        g = torch.Generator().manual_seed(0)
        low_rank = (torch.randn(8, 2, generator=g, dtype=dtype)
                    @ torch.randn(2, 8, generator=g, dtype=dtype))
        graded = torch.diag(torch.tensor([1.0, 0.5, 0.2, 0.1, 1e-3, 1e-4, 1e-5, 1e-6],
                                         dtype=dtype))
        return low_rank, graded

    @pytest.mark.parametrize('dtype, scale', [
        (torch.float32, 1e-30), (torch.float32, 1e20),
        (torch.float64, 1e-200), (torch.float64, 1e160),
    ])
    @pytest.mark.parametrize('kwargs', [
        {'rtol': 0.0}, {'rtol': 0.01}, {'cum_percentage': 0.9}, {'cum_percentage': 1.0},
    ])
    def test_truncated_svd_rtol_keeps_the_unit_scale_rank_where_squares_would_not_fit(
            self, dtype, scale, kwargs):
        """Scales past those where s² underflows (float32 below about 1e-22,
        float64 below 1e-170) or overflows (above 1e19, 1e155): the rank of the
        matrix at scale 1."""
        for matrix in self._rtol_matrices(dtype):
            assert tk.utils.truncated_svd(matrix * scale, **kwargs)[1].shape[-1] == \
                tk.utils.truncated_svd(matrix, **kwargs)[1].shape[-1]

    def test_truncated_svd_rtol_scales_each_row_by_its_own_largest(self):
        """A batch [A, A · 2⁻⁷⁰] in float32 (graded A, rtol 0.01) keeps A's rank, 3:
        the second row is scaled by its own largest value, not the batch's
        (with which its values would all count as eps, and the rank would be 8)."""
        _, graded = self._rtol_matrices(torch.float32)
        batch = torch.stack([graded, graded * 2.0 ** -70])
        assert tk.utils.truncated_svd(batch, rtol=0.01)[1].shape[-1] == 3
        assert tk.utils.truncated_svd(graded, rtol=0.01)[1].shape[-1] == 3
