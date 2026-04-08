"""
mesh.py
Builds the Cartesian grid with ghost cells.
"""

import numpy as np

def build_grid(cfg):
    """Build grid arrays from Config."""
    ng = 2
    nx_tot = cfg.nx + 2 * ng
    ny_tot = cfg.ny + 2 * ng

    x = cfg.x_min + (np.arange(nx_tot) - ng + 0.5) * cfg.dx
    y = cfg.y_min + (np.arange(ny_tot) - ng + 0.5) * cfg.dy
    X, Y = np.meshgrid(x, y, indexing='ij')

    return x, y, X, Y, nx_tot, ny_tot, ng
