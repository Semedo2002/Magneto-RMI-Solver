"""
config.py
Configuration dataclass, variable indices, and physical constants.
"""

from dataclasses import dataclass, field
from typing import List
from enum import IntEnum

# ============================================================
# Variable Indices
# ============================================================
class ConsVar(IntEnum):
    RHO = 0; MX = 1; MY = 2; MZ = 3
    BX = 4; BY = 5; BZ = 6; EN = 7
    PSI = 8; RHOC = 9

class PrimVar(IntEnum):
    RHO = 0; VX = 1; VY = 2; VZ = 3
    BX = 4; BY = 5; BZ = 6; PR = 7
    PSI = 8; CLR = 9

NVAR = 10

# shortcut indices for readability
RHO, MX, MY, MZ, BX, BY, BZ, EN, PSI, RHOC = (
    ConsVar.RHO, ConsVar.MX, ConsVar.MY, ConsVar.MZ,
    ConsVar.BX, ConsVar.BY, ConsVar.BZ, ConsVar.EN,
    ConsVar.PSI, ConsVar.RHOC)

iRHO, iVX, iVY, iVZ, iBX, iBY, iBZ, iPR, iPSI, iCLR = (
    PrimVar.RHO, PrimVar.VX, PrimVar.VY, PrimVar.VZ,
    PrimVar.BX, PrimVar.BY, PrimVar.BZ, PrimVar.PR,
    PrimVar.PSI, PrimVar.CLR)

FLOOR_RHO = 1e-12
FLOOR_PR = 1e-12


# ============================================================
# Configuration
# ============================================================
@dataclass
class Config:
    """Simulation configuration with all physical and numerical parameters."""
    nx: int = 400
    ny: int = 200
    x_min: float = 0.0
    x_max: float = 6.0
    y_min: float = 0.0
    y_max: float = 2.0
    t_end: float = 0.25
    cfl: float = 0.30
    max_steps: int = 200000
    gamma: float = 5.0 / 3.0
    mach: float = 10.0
    interface_x: float = 1.5
    perturbation_amp: float = 0.15
    perturbation_mode: int = 4
    density_ratio: float = 3.0
    interface_width: float = 2.0
    B_transverse: float = 0.0
    glm_ch: float = 0.0
    glm_alpha: float = 1.5
    powell_source: bool = True
    use_char_bc: bool = True
    bc_x_type: str = "auto"
    bc_y_type: str = "periodic"
    enable_smoothing: bool = False
    diag_interval: int = 20
    snapshot_times: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25])

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.nx

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / self.ny

    def get_bc_x(self) -> str:
        if self.bc_x_type != "auto":
            return self.bc_x_type
        return "characteristic" if self.use_char_bc else "extrapolation"
