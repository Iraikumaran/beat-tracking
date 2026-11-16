import numpy as np


def fold(bpms):
    r = np.log2(bpms / np.median(bpms))
    return (r + 0.5) % 1.0 - 0.5


def detect_change(times, bpms, folded=True, min_seg=3):
    r = fold(bpms) if folded else np.log2(bpms)
    costs = [((r[:k] - np.median(r[:k])) ** 2).sum() + ((r[k:] - np.median(r[k:])) ** 2).sum()
             for k in range(min_seg, len(r) - min_seg + 1)]
    k = min_seg + int(np.argmin(costs))
    return 0.5 * (times[k - 1] + times[k])
