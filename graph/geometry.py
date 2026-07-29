# geometry.py
"""
Geometry and probability primitives.

geometry_from_coords() computes the paper's spatio-temporal attributes
(d_Ho, theta, e) from raw 3D coordinates (Fig. 6 of the paper):
    d_Ho  : distance between hand and object of interest
    theta : angle at the hand between the hand->object vector and the
            hand->camera vector
    e     : edge = d_H / d_Ho, where d_H is the camera-to-hand distance

p_distance / p_angular / p_edge / action_score reproduce the paper's
Appendix A/B/C probability functions exactly (Gaussian, wrapped-normal,
log-normal) and its Eq. 10 action-selection rule.
"""

import math
import numpy as np


def geometry_from_coords(hand_xyz, object_xyz, camera_xyz=(0.0, 0.0, 0.0)):
    """
    hand_xyz, object_xyz, camera_xyz: (x, y, z) tuples/arrays in world coords (METERS).
    Returns dict {d_Ho, d_H, theta, e}.
    """
    hand = np.asarray(hand_xyz, dtype=float)
    obj = np.asarray(object_xyz, dtype=float)
    cam = np.asarray(camera_xyz, dtype=float)

    v_ho = obj - hand          # hand -> object
    v_hc = cam - hand          # hand -> camera

    d_ho = float(np.linalg.norm(v_ho))
    d_h = float(np.linalg.norm(hand - cam))

    denom = (np.linalg.norm(v_ho) * np.linalg.norm(v_hc))
    if denom == 0 or d_ho == 0:
        theta = 0.0
    else:
        cos_theta = np.clip(np.dot(v_ho, v_hc) / denom, -1.0, 1.0)
        theta = float(np.arccos(cos_theta))

    e = d_h / d_ho if d_ho > 1e-9 else float("inf")

    return {"d_Ho": d_ho, "d_H": d_h, "theta": theta, "e": e}


# --------------------------------------------------------------------- #
# Appendix A/B/C probability functions (unchanged from the paper)
# --------------------------------------------------------------------- #

# In graph/geometry.py, replace the three PDF functions with these:

def p_distance(d_ho, mu, sigma2):
    """Gaussian distance preference P(d_Ho) -- Appendix A."""
    # FIX: Add a variance floor to prevent mathematical collapse
    if sigma2 is None or sigma2 <= 0:
        sigma2 = 0.05 
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(
        -0.5 * ((d_ho - mu) ** 2) / sigma2
    )

def p_angular(theta, mu, sigma2, K=5):
    """Wrapped-normal angular preference P(theta) -- Appendix B."""
    # FIX: Add a variance floor
    if sigma2 is None or sigma2 <= 0:
        sigma2 = 0.05
    s = 1.0
    for k in range(1, K + 1):
        s += 2 * math.exp(-sigma2 * (k ** 2) / 2) * math.cos(k * (theta - mu))
    return max(s, 0.0) / (2 * math.pi)

def p_edge(e, mu, sigma2):
    """Log-normal edge preference P(e) -- Appendix C."""
    if e is None or e <= 0:
        return 0.0
    # FIX: Add a variance floor
    if sigma2 is None or sigma2 <= 0:
        sigma2 = 0.05
    return (1.0 / (e * math.sqrt(2 * math.pi * sigma2))) * math.exp(
        -((math.log(e) - mu) ** 2) / (2 * sigma2)
    )

def geometry_score(d_ho, theta, e, params, threshold_m=0.20):
    """
    Eq. 10:
        P(ai) = P(e)*P(theta)      if d_Ho > 0.20m (20cm)
               P(d_Ho)*P(theta)    if d_Ho <= 0.20m (20cm)
    """
    p_theta = p_angular(theta, params.get("mu_theta"), params.get("sigma2_theta"))
    
    # FIX: Compare against 0.20 meters, not 20.0 (since coordinates are in meters)
    if d_ho > threshold_m:
        return p_edge(e, params.get("mu_e"), params.get("sigma2_e")) * p_theta
    else:
        return p_distance(d_ho, params.get("mu_dHo"), params.get("sigma2_dHo")) * p_theta