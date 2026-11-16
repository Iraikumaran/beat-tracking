def acc1(est, ref, tol=0.04):
    return abs(est - ref) <= tol * ref


def acc2(est, ref, tol=0.04):
    return any(abs(est - f * ref) <= tol * f * ref for f in (1, 2, 3, 0.5, 1 / 3))


def octave_label(est, ref, tol=0.04):
    for name, f in (("correct", 1), ("double", 2), ("half", 0.5), ("triple", 3), ("third", 1 / 3)):
        if abs(est - f * ref) <= tol * f * ref:
            return name
    return "other"
