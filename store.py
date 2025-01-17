def stablemax(x: Tensor) -> Tensor:
    """
    StableMax Activation Function: A numerically stable alternative to Softmax.
    Applies piecewise linear scaling to logits to prevent numerical instability.
    
    Args:
        x (Tensor): Input tensor of logits.

    Returns:
        Tensor: Output tensor with StableMax applied.
    """
    positive_mask = x >= 0
    negative_mask = ~positive_mask

    stable_positive = (x + 1) * positive_mask
    stable_negative = (1 / (1 - x)) * negative_mask

    scaled_x = stable_positive + stable_negative
    return scaled_x / scaled_x.sum(axis=-1, keepdim=True)
