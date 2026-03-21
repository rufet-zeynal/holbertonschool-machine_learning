#!/usr/bin/env python3
"""Early Stopping"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determining if gradient descent should stop early
    cost:      current validation cost
    opt_cost:  lowest recorded validation cost
    threshold: threshold for early stopping
    patience:  patience count
    count:     how long threshold has not been met
    Returns:   (should_stop, updated_count)
    """
    if opt_cost - cost > threshold:
        # cost improved enough → reset count
        count = 0
    else:
        # cost did not improve enough → increment count
        count += 1

    # stop if count reached patience
    if count >= patience:
        return True, count

    return False, count
