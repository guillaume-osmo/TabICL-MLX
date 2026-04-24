"""FullSupportBarDistribution mean decoding (regression).

Minimal port — takes the logits from TabPFNV2p6 and the borders from the
checkpoint's ``criterion.borders`` buffer, returns a scalar prediction per
test sample.

FullSupport adds half-normal tails to the first and last buckets, and
clamps extreme mass accordingly. See tabpfn/architectures/base/bar_distribution.py
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def bar_distribution_mean(logits: mx.array, borders: mx.array) -> mx.array:
    """Simple (non-full-support) mean — softmax × bucket centres.

    Used as a baseline; FullSupport variant below is what the regressor uses.

    logits  : (..., num_buckets)
    borders : (num_buckets + 1,)
    returns : (...,)
    """
    bucket_widths = borders[1:] - borders[:-1]
    bucket_centers = borders[:-1] + bucket_widths / 2
    probs = mx.softmax(logits, axis=-1)
    return mx.sum(probs * bucket_centers, axis=-1)


def full_support_bar_distribution_mean(logits: mx.array, borders: mx.array) -> mx.array:
    """FullSupportBarDistribution mean.

    The first/last buckets are augmented with half-normal tails with std =
    bucket width. The half-normal mean offset is ``std * sqrt(2/pi)``.

    logits  : (..., num_buckets)
    borders : (num_buckets + 1,)
    """
    bucket_widths = borders[1:] - borders[:-1]
    bucket_centers = borders[:-1] + bucket_widths / 2

    # Replace first and last centers with half-normal-adjusted positions
    half_normal_factor = (2.0 / mx.pi) ** 0.5
    # Left tail: mean sits to the LEFT of border[0], shifted by std*sqrt(2/pi).
    # Std of the half-normal is set to the first bucket's width.
    left_mean = borders[0] - bucket_widths[0] * half_normal_factor
    right_mean = borders[-1] + bucket_widths[-1] * half_normal_factor

    # Replace index 0 and -1 of bucket_centers
    centers = mx.concatenate([
        left_mean.reshape(1),
        bucket_centers[1:-1],
        right_mean.reshape(1),
    ], axis=0)

    probs = mx.softmax(logits, axis=-1)
    return mx.sum(probs * centers, axis=-1)
