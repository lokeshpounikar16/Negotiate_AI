"""Small helper functions used across negotiation workflow modules."""


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))
