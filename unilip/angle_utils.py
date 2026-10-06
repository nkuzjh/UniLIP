"""Small angle helpers shared by evaluation code."""


def shortest_angle_distance(difference, period):
    """Return the nonnegative shortest circular distance for a scalar or tensor.

    Modulo first so predictions outside one turn remain valid. Both Python
    numbers and torch tensors implement these arithmetic operations.
    """
    return abs((difference + period / 2) % period - period / 2)
