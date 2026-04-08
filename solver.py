"""
solver.py
Core MHD solver: Numba kernels, HLLD, MUSCL, SSP-RK3, GLM+Powell,
MHDSolver class, verification tests, and convergence study.
"""

import numpy as np
import time
import sys
import warnings
from config import (
    Config, ConsVar, PrimVar, NVAR,
    RHO, MX, MY, MZ, BX, BY, BZ, EN, PSI, RHOC,
    iRHO, iVX, iVY, iVZ, iBX, iBY, iBZ, iPR, iPSI, iCLR,
    FLOOR_RHO, FLOOR_PR
)
from mesh import build_grid

warnings.filterwarnings("ignore")

# ============================================================
# Numba JIT setup
# ============================================================
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    def prange(*args):
        return range(*args)


# ============================================================
# Numba-accelerated kernels (the heavy lifting)
# ============================================================
@njit(cache=True)
def _cons_to_prim_kernel(U, W, gamma, floor_rho, floor_pr):
    """Convert conservative to primitive variables (Numba kernel)."""
    nvar, nx, ny = U.shape
    gm1 = gamma - 1.0
    for i in range(nx):
        for j in range(ny):
            rho = max(U[0, i, j], floor_rho)
            ri = 1.0 / rho
            vx = U[1, i, j] * ri
            vy = U[2, i, j] * ri
            vz = U[3, i, j] * ri
            bx = U[4, i, j]
            by = U[5, i, j]
            bz = U[6, i, j]
            en = U[7, i, j]
            psi = U[8, i, j]
            clr = U[9, i, j] * ri

            KE = 0.5 * rho * (vx*vx + vy*vy + vz*vz)
            ME = 0.5 * (bx*bx + by*by + bz*bz)
            pr = max(gm1 * (en - KE - ME), floor_pr)

            W[0, i, j] = rho
            W[1, i, j] = vx
            W[2, i, j] = vy
            W[3, i, j] = vz
            W[4, i, j] = bx
            W[5, i, j] = by
            W[6, i, j] = bz
            W[7, i, j] = pr
            W[8, i, j] = psi
            W[9, i, j] = clr


@njit(cache=True)
def _prim_to_cons_kernel(W, U, gamma, floor_rho, floor_pr):
    """Convert primitive to conservative variables (Numba kernel)."""
    nvar, nx, ny = W.shape
    gm1_inv = 1.0 / (gamma - 1.0)
    for i in range(nx):
        for j in range(ny):
            rho = max(W[0, i, j], floor_rho)
            vx = W[1, i, j]
            vy = W[2, i, j]
            vz = W[3, i, j]
            bx = W[4, i, j]
            by = W[5, i, j]
            bz = W[6, i, j]
            pr = max(W[7, i, j], floor_pr)

            KE = 0.5 * rho * (vx*vx + vy*vy + vz*vz)
            ME = 0.5 * (bx*bx + by*by + bz*bz)

            U[0, i, j] = rho
            U[1, i, j] = rho * vx
            U[2, i, j] = rho * vy
            U[3, i, j] = rho * vz
            U[4, i, j] = bx
            U[5, i, j] = by
            U[6, i, j] = bz
            U[7, i, j] = pr * gm1_inv + KE + ME
            U[8, i, j] = W[8, i, j]
            U[9, i, j] = rho * W[9, i, j]


@njit(cache=True)
def _vanleer_kernel(a, b):
    """Van Leer slope limiter."""
    ab = a * b
    if ab > 0.0:
        return 2.0 * ab / (a + b + 1e-30)
    return 0.0


@njit(cache=True)
def _muscl_x_kernel(W, WL, WR, nvar, nx, ny):
    """MUSCL reconstruction in x-direction (Numba kernel)."""
    for v in range(nvar):
        for i in range(1, nx - 2):
            for j in range(ny):
                dm = W[v, i, j] - W[v, i-1, j]
                dp = W[v, i+1, j] - W[v, i, j]
                d = _vanleer_kernel(dm, dp)

                dm1 = W[v, i+1, j] - W[v, i, j]
                dp1 = W[v, i+2, j] - W[v, i+1, j]
                d1 = _vanleer_kernel(dm1, dp1)

                WL[v, i-1, j] = W[v, i, j] + 0.5 * d
                WR[v, i-1, j] = W[v, i+1, j] - 0.5 * d1


@njit(cache=True)
def _muscl_y_kernel(W, WL, WR, nvar, nx, ny):
    """MUSCL reconstruction in y-direction (Numba kernel)."""
    for v in range(nvar):
        for i in range(nx):
            for j in range(1, ny - 2):
                dm = W[v, i, j] - W[v, i, j-1]
                dp = W[v, i, j+1] - W[v, i, j]
                d = _vanleer_kernel(dm, dp)

                dm1 = W[v, i, j+1] - W[v, i, j]
                dp1 = W[v, i, j+2] - W[v, i, j+1]
                d1 = _vanleer_kernel(dm1, dp1)

                WL[v, i, j-1] = W[v, i, j] + 0.5 * d
                WR[v, i, j-1] = W[v, i, j+1] - 0.5 * d1


@njit(cache=True)
def _hlld_flux_x_kernel(WL, WR, F, gamma, ch, n_faces, floor_rho, floor_pr):
    """HLLD Riemann solver for x-direction interfaces (Numba kernel)."""
    gm1 = gamma - 1.0
    smax = 0.0

    for idx in range(n_faces):
        rhoL = max(WL[0, idx], floor_rho)
        rhoR = max(WR[0, idx], floor_rho)
        vxL = WL[1, idx]; vyL = WL[2, idx]; vzL = WL[3, idx]
        vxR = WR[1, idx]; vyR = WR[2, idx]; vzR = WR[3, idx]
        BxL = WL[4, idx]; ByL = WL[5, idx]; BzL = WL[6, idx]
        BxR = WR[4, idx]; ByR = WR[5, idx]; BzR = WR[6, idx]
        pL = max(WL[7, idx], floor_pr)
        pR = max(WR[7, idx], floor_pr)
        psiL = WL[8, idx]; psiR = WR[8, idx]
        CL_v = WL[9, idx]; CR_v = WR[9, idx]

        # GLM Bx/psi
        if ch > 0:
            Bx_f = 0.5 * (BxL + BxR) - 0.5 / ch * (psiR - psiL)
            psi_f = 0.5 * (psiL + psiR) - 0.5 * ch * (BxR - BxL)
        else:
            Bx_f = 0.5 * (BxL + BxR)
            psi_f = 0.0
        Bx2 = Bx_f * Bx_f

        BmL2 = Bx2 + ByL*ByL + BzL*BzL
        BmR2 = Bx2 + ByR*ByR + BzR*BzR
        ptL = pL + 0.5 * BmL2
        ptR = pR + 0.5 * BmR2

        aL2 = gamma * pL / rhoL
        aR2 = gamma * pR / rhoR

        # Fast magnetosonic speeds
        tmpL = aL2 + BmL2/rhoL
        discL = tmpL*tmpL - 4.0*aL2*Bx2/rhoL
        if discL < 0: discL = 0.0
        cfL_sq = 0.5 * (tmpL + discL**0.5)
        if cfL_sq < 0: cfL_sq = 0.0
        cfL = cfL_sq**0.5

        tmpR = aR2 + BmR2/rhoR
        discR = tmpR*tmpR - 4.0*aR2*Bx2/rhoR
        if discR < 0: discR = 0.0
        cfR_sq = 0.5 * (tmpR + discR**0.5)
        if cfR_sq < 0: cfR_sq = 0.0
        cfR = cfR_sq**0.5

        SL = min(vxL - cfL, vxR - cfR)
        SR = max(vxL + cfL, vxR + cfR)
        if ch > 0:
            SL = min(SL, -ch)
            SR = max(SR, ch)

        abs_SL = abs(SL)
        abs_SR = abs(SR)
        if abs_SL > smax: smax = abs_SL
        if abs_SR > smax: smax = abs_SR

        dSL = SL - vxL
        dSR = SR - vxR
        den = dSR * rhoR - dSL * rhoL
        if abs(den) < 1e-14:
            if den >= 0: den = 1e-14
            else: den = -1e-14

        SM = (dSR*rhoR*vxR - dSL*rhoL*vxL - ptR + ptL) / den
        if SM < SL + 1e-14: SM = SL + 1e-14
        if SM > SR - 1e-14: SM = SR - 1e-14

        ptS = ptL + rhoL * dSL * (SM - vxL)

        # Star states
        dSL_SM = SL - SM
        if abs(dSL_SM) < 1e-14: dSL_SM = 1e-14
        rhoLS = max(rhoL * dSL / dSL_SM, floor_rho)

        dSR_SM = SR - SM
        if abs(dSR_SM) < 1e-14: dSR_SM = 1e-14
        rhoRS = max(rhoR * dSR / dSR_SM, floor_rho)

        sqLS = rhoLS**0.5
        sqRS = rhoRS**0.5

        eps_rel = 1e-8 * max(abs(ptS), 1.0)
        dnL = rhoL * dSL * dSL_SM - Bx2
        dnR = rhoR * dSR * dSR_SM - Bx2
        if abs(dnL) < eps_rel:
            if dnL >= 0: dnL = eps_rel
            else: dnL = -eps_rel
        if abs(dnR) < eps_rel:
            if dnR >= 0: dnR = eps_rel
            else: dnR = -eps_rel

        vyLS = vyL - Bx_f * ByL * (SM - vxL) / dnL
        vzLS = vzL - Bx_f * BzL * (SM - vxL) / dnL
        ByLS = ByL * (rhoL * dSL*dSL - Bx2) / dnL
        BzLS = BzL * (rhoL * dSL*dSL - Bx2) / dnL

        vyRS = vyR - Bx_f * ByR * (SM - vxR) / dnR
        vzRS = vzR - Bx_f * BzR * (SM - vxR) / dnR
        ByRS = ByR * (rhoR * dSR*dSR - Bx2) / dnR
        BzRS = BzR * (rhoR * dSR*dSR - Bx2) / dnR

        vBL = vxL*Bx_f + vyL*ByL + vzL*BzL
        vBLS = SM*Bx_f + vyLS*ByLS + vzLS*BzLS
        EL = pL/gm1 + 0.5*rhoL*(vxL*vxL + vyL*vyL + vzL*vzL) + 0.5*BmL2
        ELS = ((SL-vxL)*EL - ptL*vxL + ptS*SM + Bx_f*(vBL-vBLS)) / dSL_SM

        vBR = vxR*Bx_f + vyR*ByR + vzR*BzR
        vBRS = SM*Bx_f + vyRS*ByRS + vzRS*BzRS
        ER = pR/gm1 + 0.5*rhoR*(vxR*vxR + vyR*vyR + vzR*vzR) + 0.5*BmR2
        ERS = ((SR-vxR)*ER - ptR*vxR + ptS*SM + Bx_f*(vBR-vBRS)) / dSR_SM

        # Double star states
        sBx = 1.0 if Bx_f >= 0 else -1.0
        sqLS_safe = max(sqLS, 1e-14)
        sqRS_safe = max(sqRS, 1e-14)
        SAL = SM - abs(Bx_f) / sqLS_safe
        SAR = SM + abs(Bx_f) / sqRS_safe
        if SAL < SL: SAL = SL
        if SAL > SM: SAL = SM
        if SAR < SM: SAR = SM
        if SAR > SR: SAR = SR

        dds = max(sqLS + sqRS, 1e-14)
        vyDS = (sqLS*vyLS + sqRS*vyRS + (ByRS-ByLS)*sBx) / dds
        vzDS = (sqLS*vzLS + sqRS*vzRS + (BzRS-BzLS)*sBx) / dds
        ByDS = (sqLS*ByRS + sqRS*ByLS + sqLS*sqRS*(vyRS-vyLS)*sBx) / dds
        BzDS = (sqLS*BzRS + sqRS*BzLS + sqLS*sqRS*(vzRS-vzLS)*sBx) / dds
        vBDS = SM*Bx_f + vyDS*ByDS + vzDS*BzDS
        ELDS = ELS - sqLS * (vBLS - vBDS) * sBx
        ERDS = ERS + sqRS * (vBRS - vBDS) * sBx

        # Select flux based on wave speeds
        if SL >= 0:
            F[0, idx] = rhoL * vxL
            F[1, idx] = rhoL * vxL*vxL + ptL - Bx2
            F[2, idx] = rhoL * vyL*vxL - Bx_f*ByL
            F[3, idx] = rhoL * vzL*vxL - Bx_f*BzL
            F[4, idx] = psi_f if ch > 0 else 0.0
            F[5, idx] = ByL*vxL - Bx_f*vyL
            F[6, idx] = BzL*vxL - Bx_f*vzL
            F[7, idx] = (EL + ptL)*vxL - Bx_f*vBL
            F[8, idx] = Bx_f * ch*ch if ch > 0 else 0.0
            F[9, idx] = rhoL * CL_v * vxL
        elif SR <= 0:
            F[0, idx] = rhoR * vxR
            F[1, idx] = rhoR * vxR*vxR + ptR - Bx2
            F[2, idx] = rhoR * vyR*vxR - Bx_f*ByR
            F[3, idx] = rhoR * vzR*vxR - Bx_f*BzR
            F[4, idx] = psi_f if ch > 0 else 0.0
            F[5, idx] = ByR*vxR - Bx_f*vyR
            F[6, idx] = BzR*vxR - Bx_f*vzR
            F[7, idx] = (ER + ptR)*vxR - Bx_f*vBR
            F[8, idx] = Bx_f * ch*ch if ch > 0 else 0.0
            F[9, idx] = rhoR * CR_v * vxR
        else:
            FL0 = rhoL * vxL
            FL1 = rhoL * vxL*vxL + ptL - Bx2
            FL2 = rhoL * vyL*vxL - Bx_f*ByL
            FL3 = rhoL * vzL*vxL - Bx_f*BzL
            FL4 = psi_f if ch > 0 else 0.0
            FL5 = ByL*vxL - Bx_f*vyL
            FL6 = BzL*vxL - Bx_f*vzL
            FL7 = (EL + ptL)*vxL - Bx_f*vBL
            FL8 = Bx_f * ch*ch if ch > 0 else 0.0
            FL9 = rhoL * CL_v * vxL

            UL0 = rhoL; UL1 = rhoL*vxL; UL2 = rhoL*vyL; UL3 = rhoL*vzL
            UL4 = Bx_f; UL5 = ByL; UL6 = BzL; UL7 = EL; UL8 = psiL; UL9 = rhoL*CL_v

            ULS0 = rhoLS; ULS1 = rhoLS*SM; ULS2 = rhoLS*vyLS; ULS3 = rhoLS*vzLS
            ULS4 = Bx_f; ULS5 = ByLS; ULS6 = BzLS; ULS7 = ELS; ULS8 = psi_f; ULS9 = rhoLS*CL_v

            FR0 = rhoR * vxR
            FR1 = rhoR * vxR*vxR + ptR - Bx2
            FR2 = rhoR * vyR*vxR - Bx_f*ByR
            FR3 = rhoR * vzR*vxR - Bx_f*BzR
            FR4 = psi_f if ch > 0 else 0.0
            FR5 = ByR*vxR - Bx_f*vyR
            FR6 = BzR*vxR - Bx_f*vzR
            FR7 = (ER + ptR)*vxR - Bx_f*vBR
            FR8 = Bx_f * ch*ch if ch > 0 else 0.0
            FR9 = rhoR * CR_v * vxR

            UR0 = rhoR; UR1 = rhoR*vxR; UR2 = rhoR*vyR; UR3 = rhoR*vzR
            UR4 = Bx_f; UR5 = ByR; UR6 = BzR; UR7 = ER; UR8 = psiR; UR9 = rhoR*CR_v

            URS0 = rhoRS; URS1 = rhoRS*SM; URS2 = rhoRS*vyRS; URS3 = rhoRS*vzRS
            URS4 = Bx_f; URS5 = ByRS; URS6 = BzRS; URS7 = ERS; URS8 = psi_f; URS9 = rhoRS*CR_v

            FLS0 = FL0 + SL*(ULS0-UL0); FLS1 = FL1 + SL*(ULS1-UL1)
            FLS2 = FL2 + SL*(ULS2-UL2); FLS3 = FL3 + SL*(ULS3-UL3)
            FLS4 = FL4 + SL*(ULS4-UL4); FLS5 = FL5 + SL*(ULS5-UL5)
            FLS6 = FL6 + SL*(ULS6-UL6); FLS7 = FL7 + SL*(ULS7-UL7)
            FLS8 = FL8 + SL*(ULS8-UL8); FLS9 = FL9 + SL*(ULS9-UL9)

            ULDS0 = rhoLS; ULDS1 = rhoLS*SM; ULDS2 = rhoLS*vyDS; ULDS3 = rhoLS*vzDS
            ULDS4 = Bx_f; ULDS5 = ByDS; ULDS6 = BzDS; ULDS7 = ELDS; ULDS8 = psi_f; ULDS9 = rhoLS*CL_v

            FLDS0 = FLS0 + SAL*(ULDS0-ULS0); FLDS1 = FLS1 + SAL*(ULDS1-ULS1)
            FLDS2 = FLS2 + SAL*(ULDS2-ULS2); FLDS3 = FLS3 + SAL*(ULDS3-ULS3)
            FLDS4 = FLS4 + SAL*(ULDS4-ULS4); FLDS5 = FLS5 + SAL*(ULDS5-ULS5)
            FLDS6 = FLS6 + SAL*(ULDS6-ULS6); FLDS7 = FLS7 + SAL*(ULDS7-ULS7)
            FLDS8 = FLS8 + SAL*(ULDS8-ULS8); FLDS9 = FLS9 + SAL*(ULDS9-ULS9)

            FRS0 = FR0 + SR*(URS0-UR0); FRS1 = FR1 + SR*(URS1-UR1)
            FRS2 = FR2 + SR*(URS2-UR2); FRS3 = FR3 + SR*(URS3-UR3)
            FRS4 = FR4 + SR*(URS4-UR4); FRS5 = FR5 + SR*(URS5-UR5)
            FRS6 = FR6 + SR*(URS6-UR6); FRS7 = FR7 + SR*(URS7-UR7)
            FRS8 = FR8 + SR*(URS8-UR8); FRS9 = FR9 + SR*(URS9-UR9)

            URDS0 = rhoRS; URDS1 = rhoRS*SM; URDS2 = rhoRS*vyDS; URDS3 = rhoRS*vzDS
            URDS4 = Bx_f; URDS5 = ByDS; URDS6 = BzDS; URDS7 = ERDS; URDS8 = psi_f; URDS9 = rhoRS*CR_v

            FRDS0 = FRS0 + SAR*(URDS0-URS0); FRDS1 = FRS1 + SAR*(URDS1-URS1)
            FRDS2 = FRS2 + SAR*(URDS2-URS2); FRDS3 = FRS3 + SAR*(URDS3-URS3)
            FRDS4 = FRS4 + SAR*(URDS4-URS4); FRDS5 = FRS5 + SAR*(URDS5-URS5)
            FRDS6 = FRS6 + SAR*(URDS6-URS6); FRDS7 = FRS7 + SAR*(URDS7-URS7)
            FRDS8 = FRS8 + SAR*(URDS8-URS8); FRDS9 = FRS9 + SAR*(URDS9-URS9)

            if SAL >= 0:
                F[0,idx]=FLS0; F[1,idx]=FLS1; F[2,idx]=FLS2; F[3,idx]=FLS3
                F[4,idx]=FLS4; F[5,idx]=FLS5; F[6,idx]=FLS6; F[7,idx]=FLS7
                F[8,idx]=FLS8; F[9,idx]=FLS9
            elif SM >= 0:
                F[0,idx]=FLDS0; F[1,idx]=FLDS1; F[2,idx]=FLDS2; F[3,idx]=FLDS3
                F[4,idx]=FLDS4; F[5,idx]=FLDS5; F[6,idx]=FLDS6; F[7,idx]=FLDS7
                F[8,idx]=FLDS8; F[9,idx]=FLDS9
            elif SAR >= 0:
                F[0,idx]=FRDS0; F[1,idx]=FRDS1; F[2,idx]=FRDS2; F[3,idx]=FRDS3
                F[4,idx]=FRDS4; F[5,idx]=FRDS5; F[6,idx]=FRDS6; F[7,idx]=FRDS7
                F[8,idx]=FRDS8; F[9,idx]=FRDS9
            else:
                F[0,idx]=FRS0; F[1,idx]=FRS1; F[2,idx]=FRS2; F[3,idx]=FRS3
                F[4,idx]=FRS4; F[5,idx]=FRS5; F[6,idx]=FRS6; F[7,idx]=FRS7
                F[8,idx]=FRS8; F[9,idx]=FRS9

            # GLM override
            if ch > 0:
                F[4, idx] = psi_f
                F[8, idx] = ch*ch * Bx_f

    return smax


@njit(cache=True)
def _powell_source_face_kernel(W, S, dx, dy, nvar, nx, ny):
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            Bx_face_ip = 0.5 * (W[4, i, j] + W[4, i+1, j])
            Bx_face_im = 0.5 * (W[4, i-1, j] + W[4, i, j])
            By_face_jp = 0.5 * (W[5, i, j] + W[5, i, j+1])
            By_face_jm = 0.5 * (W[5, i, j-1] + W[5, i, j])

            divB = (Bx_face_ip - Bx_face_im) / dx + (By_face_jp - By_face_jm) / dy

            rho = W[0, i, j]
            vx = W[1, i, j]
            vy = W[2, i, j]
            vz = W[3, i, j]
            bx = W[4, i, j]
            by = W[5, i, j]
            bz = W[6, i, j]
            vB = vx*bx + vy*by + vz*bz

            S[0, i, j] = 0.0
            S[1, i, j] = -divB * bx
            S[2, i, j] = -divB * by
            S[3, i, j] = -divB * bz
            S[4, i, j] = -divB * vx
            S[5, i, j] = -divB * vy
            S[6, i, j] = -divB * vz
            S[7, i, j] = -divB * vB
            S[8, i, j] = 0.0
            S[9, i, j] = 0.0


# ============================================================
# NumPy fallback versions (used when Numba unavailable)
# ============================================================
def cons_to_prim(U, gamma):
    """Convert conservative to primitive variables."""
    if HAS_NUMBA:
        W = np.empty_like(U)
        _cons_to_prim_kernel(U, W, gamma, FLOOR_RHO, FLOOR_PR)
        return W
    W = np.empty_like(U)
    rho = np.maximum(U[RHO], FLOOR_RHO)
    ri = 1.0 / rho
    W[iRHO] = rho
    W[iVX] = U[MX] * ri
    W[iVY] = U[MY] * ri
    W[iVZ] = U[MZ] * ri
    W[iBX] = U[BX]
    W[iBY] = U[BY]
    W[iBZ] = U[BZ]
    W[iPSI] = U[PSI]
    W[iCLR] = U[RHOC] * ri
    KE = 0.5 * rho * (W[iVX]**2 + W[iVY]**2 + W[iVZ]**2)
    ME = 0.5 * (U[BX]**2 + U[BY]**2 + U[BZ]**2)
    W[iPR] = np.maximum((gamma - 1) * (U[EN] - KE - ME), FLOOR_PR)
    return W


def prim_to_cons(W, gamma):
    """Convert primitive to conservative variables."""
    if HAS_NUMBA:
        U = np.empty_like(W)
        _prim_to_cons_kernel(W, U, gamma, FLOOR_RHO, FLOOR_PR)
        return U
    U = np.empty_like(W)
    rho = np.maximum(W[iRHO], FLOOR_RHO)
    U[RHO] = rho
    U[MX] = rho * W[iVX]
    U[MY] = rho * W[iVY]
    U[MZ] = rho * W[iVZ]
    U[BX] = W[iBX]
    U[BY] = W[iBY]
    U[BZ] = W[iBZ]
    U[PSI] = W[iPSI]
    U[RHOC] = rho * W[iCLR]
    KE = 0.5 * rho * (W[iVX]**2 + W[iVY]**2 + W[iVZ]**2)
    ME = 0.5 * (W[iBX]**2 + W[iBY]**2 + W[iBZ]**2)
    U[EN] = W[iPR] / (gamma - 1) + KE + ME
    return U


# ============================================================
# MHD Rankine-Hugoniot
# ============================================================
def mhd_rankine_hugoniot(gamma, M, rho1, p1, By1):
    """Solve MHD Rankine-Hugoniot jump conditions for a fast shock."""
    M2 = M * M
    if abs(By1) < 1e-14:
        r = (gamma + 1) * M2 / ((gamma - 1) * M2 + 2)
        cs1 = np.sqrt(gamma * p1 / rho1)
        vs = M * cs1
        p2 = p1 * (2 * gamma * M2 - (gamma - 1)) / (gamma + 1)
        vx2 = vs * (1 - 1.0 / r)
        return rho1 * r, p2, vx2, 0.0, vs

    cs1_sq = gamma * p1 / rho1
    va1_sq = By1**2 / rho1
    cf1 = np.sqrt(cs1_sq + va1_sq)
    vs = M * cf1

    def get_p2(r):
        return p1 + rho1 * vs**2 * (1 - 1.0/r) + 0.5 * By1**2 * (1 - r**2)

    def energy_residual(r):
        p2_ = get_p2(r)
        lhs = 0.5*vs**2 + gamma/(gamma-1)*p1/rho1 + By1**2/rho1
        rhs = (0.5*vs**2/r**2 + gamma/(gamma-1)*p2_/(rho1*r) + r*By1**2/rho1)
        return lhs - rhs

    r = (gamma + 1) * M2 / ((gamma - 1) * M2 + 2)
    r = min(r, (gamma + 1) / (gamma - 1) - 0.01)

    for _ in range(100):
        f = energy_residual(r)
        dr = 1e-6 * r
        fp = (energy_residual(r + dr) - energy_residual(r - dr)) / (2 * dr)
        if abs(fp) < 1e-30: break
        r_new = r - f / fp
        r_new = max(r_new, 1.001)
        r_new = min(r_new, (gamma + 1) / (gamma - 1) - 0.001)
        if abs(r_new - r) < 1e-12:
            r = r_new; break
        r = r_new

    rho2 = rho1 * r
    p2 = get_p2(r)
    By2 = r * By1
    vx2 = vs * (1 - 1.0 / r)
    return rho2, p2, vx2, By2, vs


# ============================================================
# HLLD flux wrappers
# ============================================================
def hlld_flux_x(WL, WR, gamma, ch):
    """Compute HLLD numerical flux in x-direction."""
    n_faces = WL.shape[1]
    F = np.empty_like(WL)

    if HAS_NUMBA:
        smax = _hlld_flux_x_kernel(
            np.ascontiguousarray(WL),
            np.ascontiguousarray(WR),
            F, gamma, ch, n_faces, FLOOR_RHO, FLOOR_PR)
        return F, smax

    return _hlld_flux_x_numpy(WL, WR, gamma, ch)


def _hlld_flux_x_numpy(WL, WR, gamma, ch):
    """NumPy vectorized HLLD flux (fallback when Numba unavailable)."""
    rhoL = np.maximum(WL[iRHO], FLOOR_RHO)
    rhoR = np.maximum(WR[iRHO], FLOOR_RHO)
    vxL, vyL, vzL = WL[iVX], WL[iVY], WL[iVZ]
    vxR, vyR, vzR = WR[iVX], WR[iVY], WR[iVZ]
    BxL, ByL, BzL = WL[iBX], WL[iBY], WL[iBZ]
    BxR, ByR, BzR = WR[iBX], WR[iBY], WR[iBZ]
    pL = np.maximum(WL[iPR], FLOOR_PR)
    pR = np.maximum(WR[iPR], FLOOR_PR)
    psiL, psiR = WL[iPSI], WR[iPSI]
    CL, CR = WL[iCLR], WR[iCLR]

    if ch > 0:
        Bx = 0.5*(BxL+BxR) - 0.5/ch*(psiR-psiL)
        psi_f = 0.5*(psiL+psiR) - 0.5*ch*(BxR-BxL)
    else:
        Bx = 0.5*(BxL+BxR)
        psi_f = np.zeros_like(BxL)
    Bx2 = Bx**2

    BmL2 = Bx2 + ByL**2 + BzL**2
    BmR2 = Bx2 + ByR**2 + BzR**2
    ptL = pL + 0.5*BmL2
    ptR = pR + 0.5*BmR2

    aL2 = gamma*pL/rhoL; aR2 = gamma*pR/rhoR
    cfL = np.sqrt(np.maximum(0.5*(aL2+BmL2/rhoL+np.sqrt(np.maximum((aL2+BmL2/rhoL)**2-4*aL2*Bx2/rhoL,0))),0))
    cfR = np.sqrt(np.maximum(0.5*(aR2+BmR2/rhoR+np.sqrt(np.maximum((aR2+BmR2/rhoR)**2-4*aR2*Bx2/rhoR,0))),0))

    SL = np.minimum(vxL-cfL, vxR-cfR)
    SR = np.maximum(vxL+cfL, vxR+cfR)
    if ch > 0: SL = np.minimum(SL,-ch); SR = np.maximum(SR,ch)

    dSL = SL-vxL; dSR = SR-vxR
    den = dSR*rhoR - dSL*rhoL
    den = np.where(np.abs(den)<1e-14, np.sign(den+1e-30)*1e-14, den)
    SM = (dSR*rhoR*vxR - dSL*rhoL*vxL - ptR + ptL)/den
    SM = np.clip(SM, SL+1e-14, SR-1e-14)

    ptS = ptL + rhoL*dSL*(SM-vxL)
    rhoLS = np.maximum(rhoL*dSL/np.where(np.abs(SL-SM)<1e-14,1e-14,SL-SM), FLOOR_RHO)
    rhoRS = np.maximum(rhoR*dSR/np.where(np.abs(SR-SM)<1e-14,1e-14,SR-SM), FLOOR_RHO)
    sqLS = np.sqrt(rhoLS); sqRS = np.sqrt(rhoRS)

    eps_rel = 1e-8*np.maximum(np.abs(ptS),1.0)
    dnL = rhoL*dSL*(SL-SM)-Bx2; dnR = rhoR*dSR*(SR-SM)-Bx2
    dnL = np.where(np.abs(dnL)<eps_rel, np.sign(dnL+1e-30)*eps_rel, dnL)
    dnR = np.where(np.abs(dnR)<eps_rel, np.sign(dnR+1e-30)*eps_rel, dnR)

    vyLS=vyL-Bx*ByL*(SM-vxL)/dnL; vzLS=vzL-Bx*BzL*(SM-vxL)/dnL
    ByLS=ByL*(rhoL*dSL**2-Bx2)/dnL; BzLS=BzL*(rhoL*dSL**2-Bx2)/dnL
    vyRS=vyR-Bx*ByR*(SM-vxR)/dnR; vzRS=vzR-Bx*BzR*(SM-vxR)/dnR
    ByRS=ByR*(rhoR*dSR**2-Bx2)/dnR; BzRS=BzR*(rhoR*dSR**2-Bx2)/dnR

    vBL=vxL*Bx+vyL*ByL+vzL*BzL; vBLS=SM*Bx+vyLS*ByLS+vzLS*BzLS
    EL=pL/(gamma-1)+0.5*rhoL*(vxL**2+vyL**2+vzL**2)+0.5*BmL2
    ELS=((SL-vxL)*EL-ptL*vxL+ptS*SM+Bx*(vBL-vBLS))/np.where(np.abs(SL-SM)<1e-14,1e-14,SL-SM)
    vBR=vxR*Bx+vyR*ByR+vzR*BzR; vBRS=SM*Bx+vyRS*ByRS+vzRS*BzRS
    ER=pR/(gamma-1)+0.5*rhoR*(vxR**2+vyR**2+vzR**2)+0.5*BmR2
    ERS=((SR-vxR)*ER-ptR*vxR+ptS*SM+Bx*(vBR-vBRS))/np.where(np.abs(SR-SM)<1e-14,1e-14,SR-SM)

    sBx=np.sign(Bx)
    SAL=SM-np.abs(Bx)/np.maximum(sqLS,1e-14); SAR=SM+np.abs(Bx)/np.maximum(sqRS,1e-14)
    SAL=np.maximum(SAL,SL); SAL=np.minimum(SAL,SM)
    SAR=np.maximum(SAR,SM); SAR=np.minimum(SAR,SR)
    dds=np.maximum(sqLS+sqRS,1e-14)
    vyDS=(sqLS*vyLS+sqRS*vyRS+(ByRS-ByLS)*sBx)/dds
    vzDS=(sqLS*vzLS+sqRS*vzRS+(BzRS-BzLS)*sBx)/dds
    ByDS=(sqLS*ByRS+sqRS*ByLS+sqLS*sqRS*(vyRS-vyLS)*sBx)/dds
    BzDS=(sqLS*BzRS+sqRS*BzLS+sqLS*sqRS*(vzRS-vzLS)*sBx)/dds
    vBDS=SM*Bx+vyDS*ByDS+vzDS*BzDS
    ELDS=ELS-sqLS*(vBLS-vBDS)*sBx; ERDS=ERS+sqRS*(vBRS-vBDS)*sBx

    def mk_cons(rho_v,vx_v,vy_v,vz_v,By_v,Bz_v,E_v,psi_v,C_v):
        Uv=np.empty_like(WL); Uv[RHO]=rho_v; Uv[MX]=rho_v*vx_v; Uv[MY]=rho_v*vy_v
        Uv[MZ]=rho_v*vz_v; Uv[BX]=Bx; Uv[BY]=By_v; Uv[BZ]=Bz_v; Uv[EN]=E_v
        Uv[PSI]=psi_v; Uv[RHOC]=rho_v*C_v; return Uv

    def mk_flux(rho_v,vx_v,vy_v,vz_v,By_v,Bz_v,E_v,pt_v,vB_v,psi_v,C_v):
        Fv=np.empty_like(WL); Fv[RHO]=rho_v*vx_v; Fv[MX]=rho_v*vx_v**2+pt_v-Bx2
        Fv[MY]=rho_v*vy_v*vx_v-Bx*By_v; Fv[MZ]=rho_v*vz_v*vx_v-Bx*Bz_v
        Fv[BX]=psi_v if ch>0 else np.zeros_like(Bx); Fv[BY]=By_v*vx_v-Bx*vy_v
        Fv[BZ]=Bz_v*vx_v-Bx*vz_v; Fv[EN]=(E_v+pt_v)*vx_v-Bx*vB_v
        Fv[PSI]=Bx*ch**2 if ch>0 else np.zeros_like(Bx); Fv[RHOC]=rho_v*C_v*vx_v
        return Fv

    FL=mk_flux(rhoL,vxL,vyL,vzL,ByL,BzL,EL,ptL,vBL,psi_f,CL)
    UL=mk_cons(rhoL,vxL,vyL,vzL,ByL,BzL,EL,psiL,CL)
    FR=mk_flux(rhoR,vxR,vyR,vzR,ByR,BzR,ER,ptR,vBR,psi_f,CR)
    UR=mk_cons(rhoR,vxR,vyR,vzR,ByR,BzR,ER,psiR,CR)
    ULS=mk_cons(rhoLS,SM,vyLS,vzLS,ByLS,BzLS,ELS,psi_f,CL)
    URS=mk_cons(rhoRS,SM,vyRS,vzRS,ByRS,BzRS,ERS,psi_f,CR)
    ULDS=mk_cons(rhoLS,SM,vyDS,vzDS,ByDS,BzDS,ELDS,psi_f,CL)
    URDS=mk_cons(rhoRS,SM,vyDS,vzDS,ByDS,BzDS,ERDS,psi_f,CR)

    FLS=FL+SL*(ULS-UL); FLDS=FLS+SAL*(ULDS-ULS)
    FRS=FR+SR*(URS-UR); FRDS=FRS+SAR*(URDS-URS)
    Fo=np.where(SL>=0,FL,np.where(SAL>=0,FLS,np.where(SM>=0,FLDS,np.where(SAR>=0,FRDS,np.where(SR>=0,FRS,FR)))))
    if ch>0: Fo[BX]=psi_f; Fo[PSI]=ch**2*Bx
    smax=max(float(np.max(np.abs(SL))),float(np.max(np.abs(SR))))
    return Fo,smax


def hlld_flux_y(WL, WR, gamma, ch):
    """Compute HLLD numerical flux in y-direction by coordinate rotation."""
    perm = np.arange(NVAR)
    perm[iVX], perm[iVY] = iVY, iVX
    perm[iBX], perm[iBY] = iBY, iBX
    Fr, sm = hlld_flux_x(WL[perm], WR[perm], gamma, ch)
    return Fr[perm], sm


# ============================================================
# MUSCL Reconstruction (slope-limited)
# ============================================================
def muscl_x(W):
    """MUSCL reconstruction in x with van Leer limiter."""
    if HAS_NUMBA:
        nvar, nx, ny = W.shape
        n_faces = nx - 3
        WL = np.empty((nvar, n_faces, ny))
        WR = np.empty((nvar, n_faces, ny))
        _muscl_x_kernel(np.ascontiguousarray(W), WL, WR, nvar, nx, ny)
        WL[iRHO] = np.maximum(WL[iRHO], FLOOR_RHO)
        WR[iRHO] = np.maximum(WR[iRHO], FLOOR_RHO)
        WL[iPR] = np.maximum(WL[iPR], FLOOR_PR)
        WR[iPR] = np.maximum(WR[iPR], FLOOR_PR)
        return WL, WR
    return _muscl_x_numpy(W)


def _muscl_x_numpy(W):
    """NumPy fallback for MUSCL-x."""
    dm = W[:, 1:-1, :] - W[:, :-2, :]
    dp = W[:, 2:, :] - W[:, 1:-1, :]
    ab = dm * dp
    d = np.where(ab > 0, 2.0 * ab / (dm + dp + 1e-30), 0.0)
    WL = W[:, 1:-2, :] + 0.5 * d[:, :-1, :]
    WR = W[:, 2:-1, :] - 0.5 * d[:, 1:, :]
    WL[iRHO] = np.maximum(WL[iRHO], FLOOR_RHO)
    WR[iRHO] = np.maximum(WR[iRHO], FLOOR_RHO)
    WL[iPR] = np.maximum(WL[iPR], FLOOR_PR)
    WR[iPR] = np.maximum(WR[iPR], FLOOR_PR)
    return WL, WR


def muscl_y(W):
    """MUSCL reconstruction in y with van Leer limiter."""
    if HAS_NUMBA:
        nvar, nx, ny = W.shape
        n_faces = ny - 3
        WL = np.empty((nvar, nx, n_faces))
        WR = np.empty((nvar, nx, n_faces))
        _muscl_y_kernel(np.ascontiguousarray(W), WL, WR, nvar, nx, ny)
        WL[iRHO] = np.maximum(WL[iRHO], FLOOR_RHO)
        WR[iRHO] = np.maximum(WR[iRHO], FLOOR_RHO)
        WL[iPR] = np.maximum(WL[iPR], FLOOR_PR)
        WR[iPR] = np.maximum(WR[iPR], FLOOR_PR)
        return WL, WR
    return _muscl_y_numpy(W)


def _muscl_y_numpy(W):
    """NumPy fallback for MUSCL-y."""
    dm = W[:, :, 1:-1] - W[:, :, :-2]
    dp = W[:, :, 2:] - W[:, :, 1:-1]
    ab = dm * dp
    d = np.where(ab > 0, 2.0 * ab / (dm + dp + 1e-30), 0.0)
    WL = W[:, :, 1:-2] + 0.5 * d[:, :, :-1]
    WR = W[:, :, 2:-1] - 0.5 * d[:, :, 1:]
    WL[iRHO] = np.maximum(WL[iRHO], FLOOR_RHO)
    WR[iRHO] = np.maximum(WR[iRHO], FLOOR_RHO)
    WL[iPR] = np.maximum(WL[iPR], FLOOR_PR)
    WR[iPR] = np.maximum(WR[iPR], FLOOR_PR)
    return WL, WR


def smooth(y, n=5, enabled=True):
    """Optional smoothing filter for diagnostics."""
    if not enabled:
        return np.asarray(y, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 2 * n:
        return y
    kernel = np.ones(n) / n
    yp = np.concatenate([np.full(n, y[0]), y, np.full(n, y[-1])])
    return np.convolve(yp, kernel, mode='same')[n:-n]


# ============================================================
# MHD Solver Class (the main driver)
# ============================================================
class MHDSolver:
    """2D Ideal MHD solver with HLLD, MUSCL-Hancock, SSP-RK3, GLM+Powell."""

    def __init__(self, config):
        self.cfg = config
        self.ng = 2
        self.nx_tot = config.nx + 2 * self.ng
        self.ny_tot = config.ny + 2 * self.ng

        self.x = config.x_min + (np.arange(self.nx_tot) - self.ng + 0.5) * config.dx
        self.y = config.y_min + (np.arange(self.ny_tot) - self.ng + 0.5) * config.dy
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        self.U = np.zeros((NVAR, self.nx_tot, self.ny_tot))
        self.t = 0.0
        self.step = 0
        self.dt = 0.0
        self._glm_ch_frozen = 0.0

        # Diagnostics storage
        self.diag_times = []
        self.diag_mixing_width_integral = []
        self.diag_mixing_width_thresh = []
        self.diag_mixedness = []
        self.diag_enstrophy = []
        self.diag_enstrophy_local = []
        self.diag_perturbation_amp = []
        self.diag_mode_amps = []
        self.diag_stag_pressure = []
        self.diag_divB_max = []
        self.diag_divB_L2 = []
        self.diag_energy_total = []
        self.diag_boundary_flux_cumulative = []
        self._cumulative_boundary_energy = 0.0
        self.snapshots = {}
        self._rho_light = 1.0
        self._rho_heavy = 1.0

        self._rh_results = None

    def initialize(self):
        """Set up initial conditions: shocked gas + perturbed density interface."""
        cfg = self.cfg
        g = cfg.gamma
        M = cfg.mach
        rho1 = 1.0; p1 = 1.0
        By_pre = cfg.B_transverse

        rho2, p2, vx2, By2, vs = mhd_rankine_hugoniot(g, M, rho1, p1, By_pre)
        self._rh_results = {'rho2': rho2, 'p2': p2, 'vx2': vx2, 'By2': By2, 'vs': vs}

        rho_h = rho1 * cfg.density_ratio
        x_shock = cfg.interface_x - 0.3
        Ly = cfg.y_max - cfg.y_min
        ky = 2 * np.pi * cfg.perturbation_mode / Ly

        x_if = cfg.interface_x + cfg.perturbation_amp * np.sin(ky * self.Y)
        interface_delta = cfg.interface_width * cfg.dx
        phi = 0.5 * (1.0 + np.tanh((self.X - x_if) / max(interface_delta, 1e-10)))

        W = np.zeros((NVAR, self.nx_tot, self.ny_tot))
        post = self.X < x_shock
        rho_pre = rho1 * (1 - phi) + rho_h * phi

        W[iRHO] = np.where(post, rho2, rho_pre)
        W[iVX] = np.where(post, vx2, 0.0)
        W[iBY] = np.where(post, By2, By_pre)
        W[iPR] = np.where(post, p2, p1)
        W[iCLR] = np.where(post, 0.0, phi)

        self.U = prim_to_cons(W, g)
        self.t = 0.0; self.step = 0
        self._cumulative_boundary_energy = 0.0
        self._rho_light = rho1; self._rho_heavy = rho_h

        print(f"  Mach={M:.1f}, Vs={vs:.3f}, rho2={rho2:.3f}, p2={p2:.3f}, vx2={vx2:.3f}")
        if cfg.B_transverse > 0:
            beta = 2*p1/cfg.B_transverse**2
            va = cfg.B_transverse/np.sqrt(rho1)
            print(f"  By_pre={By_pre:.3f}, By_post={By2:.3f}, beta={beta:.2f}, v_A={va:.3f}")
        else:
            print(f"  B_y=0 (pure hydro)")
        bc_x = cfg.get_bc_x()
        print(f"  Grid: {cfg.nx}x{cfg.ny}, dx={cfg.dx:.4f}, interface_width={cfg.interface_width:.1f} cells")
        if cfg.powell_source:
            print(f"  Powell source terms: ENABLED (face-centered div(B))")
        print(f"  BC x: {bc_x}, BC y: {cfg.bc_y_type}")
        sys.stdout.flush()

    def apply_bc(self, U):
        ng = self.ng
        cfg = self.cfg
        bc_x = cfg.get_bc_x()

        if bc_x == "periodic":
            U[:, :ng, :] = U[:, -2*ng:-ng, :]
            U[:, -ng:, :] = U[:, ng:2*ng, :]
        elif bc_x == "characteristic":
            self._apply_characteristic_bc_x(U)
        else:
            for i in range(ng):
                U[:, i, :] = U[:, ng, :]
                U[:, -(i+1), :] = U[:, -(ng+1), :]

        if bc_x != "periodic":
            U[PSI, :ng, :] = 0
            U[PSI, -ng:, :] = 0

        if cfg.bc_y_type == "periodic":
            U[:, :, :ng] = U[:, :, -2*ng:-ng]
            U[:, :, -ng:] = U[:, :, ng:2*ng]
        else:
            for j in range(ng):
                U[:, :, j] = U[:, :, ng]
                U[:, :, -(j+1)] = U[:, :, -(ng+1)]

        return U

    def _apply_characteristic_bc_x(self, U):
        ng = self.ng
        for i in range(ng):
            U[:, i, :] = U[:, ng, :]
            U[:, -(i+1), :] = U[:, -(ng+1), :]
        for i in range(ng):
            U[PSI, -(i+1), :] *= 0.1
            U[PSI, i, :] *= 0.1

    def enforce_scalar_bounds(self, U):
        rho = np.maximum(U[RHO], FLOOR_RHO)
        C = np.clip(U[RHOC] / rho, 0.0, 1.0)
        U[RHOC] = rho * C
        return U

    def compute_dt(self, W):
        cfg = self.cfg
        rho = np.maximum(W[iRHO], FLOOR_RHO)
        p = np.maximum(W[iPR], FLOOR_PR)
        B2 = W[iBX]**2 + W[iBY]**2 + W[iBZ]**2
        cf = np.sqrt(cfg.gamma * p / rho + B2 / rho)
        cf_max = float(np.max(cf))
        v_abs_max = float(np.max(np.abs(W[iVX]) + np.abs(W[iVY])))
        cfg.glm_ch = max(cf_max + v_abs_max, 1.0) * 1.5
        ch = cfg.glm_ch
        sx = np.abs(W[iVX]) + np.maximum(cf, ch)
        sy = np.abs(W[iVY]) + np.maximum(cf, ch)
        inv_dt = np.maximum(sx / cfg.dx, sy / cfg.dy)
        sm = float(np.max(inv_dt))
        if sm < 1e-14:
            return cfg.cfl * min(cfg.dx, cfg.dy)
        return cfg.cfl / sm

    def _compute_boundary_energy_flux(self, U):
        ng = self.ng; cfg = self.cfg
        if cfg.get_bc_x() == "periodic":
            return 0.0

        W = cons_to_prim(U, cfg.gamma)
        def boundary_flux(idx):
            rho_b=W[iRHO,idx,ng:-ng]; vx_b=W[iVX,idx,ng:-ng]
            p_b=W[iPR,idx,ng:-ng]; Bx_b=W[iBX,idx,ng:-ng]
            By_b=W[iBY,idx,ng:-ng]; Bz_b=W[iBZ,idx,ng:-ng]
            B2_b=Bx_b**2+By_b**2+Bz_b**2; pt_b=p_b+0.5*B2_b
            E_b=U[EN,idx,ng:-ng]
            vB_b=vx_b*Bx_b+W[iVY,idx,ng:-ng]*By_b+W[iVZ,idx,ng:-ng]*Bz_b
            return float(np.sum(((E_b+pt_b)*vx_b-Bx_b*vB_b)*cfg.dy))
        return boundary_flux(ng) - boundary_flux(-(ng+1))

    def _compute_powell_source(self, U, W):
        cfg = self.cfg
        S = np.zeros_like(U)

        if HAS_NUMBA:
            _powell_source_face_kernel(
                np.ascontiguousarray(W), S,
                cfg.dx, cfg.dy, NVAR, self.nx_tot, self.ny_tot)
        else:
            Bx = W[iBX]; By = W[iBY]
            divB = np.zeros_like(Bx)
            divB[1:-1, 1:-1] = (
                (0.5*(Bx[1:-1, 1:-1] + Bx[2:, 1:-1]) -
                 0.5*(Bx[:-2, 1:-1] + Bx[1:-1, 1:-1])) / cfg.dx +
                (0.5*(By[1:-1, 1:-1] + By[1:-1, 2:]) -
                 0.5*(By[1:-1, :-2] + By[1:-1, 1:-1])) / cfg.dy
            )
            vB = W[iVX]*W[iBX] + W[iVY]*W[iBY] + W[iVZ]*W[iBZ]
            S[MX] = -divB * W[iBX]
            S[MY] = -divB * W[iBY]
            S[MZ] = -divB * W[iBZ]
            S[BX] = -divB * W[iVX]
            S[BY] = -divB * W[iVY]
            S[BZ] = -divB * W[iVZ]
            S[EN] = -divB * vB
        return S

    def compute_rhs(self, U):
        cfg = self.cfg
        ng = self.ng
        nx = cfg.nx; ny = cfg.ny
        g = cfg.gamma
        ch = self._glm_ch_frozen

        U = self.apply_bc(U)
        W = cons_to_prim(U, g)

        WLx, WRx = muscl_x(W)
        s = WLx.shape
        Fx, _ = hlld_flux_x(WLx.reshape(NVAR, -1), WRx.reshape(NVAR, -1), g, ch)
        Fx = Fx.reshape(s)

        WLy, WRy = muscl_y(W)
        s2 = WLy.shape
        Fy, _ = hlld_flux_y(WLy.reshape(NVAR, -1), WRy.reshape(NVAR, -1), g, ch)
        Fy = Fy.reshape(s2)

        dFx = Fx[:, 1:1+nx, :] - Fx[:, 0:nx, :]
        dFy = Fy[:, :, 1:1+ny] - Fy[:, :, 0:ny]

        R = np.zeros_like(U)
        R[:, ng:ng+nx, ng:ng+ny] -= dFx[:, :, ng:ng+ny] / cfg.dx
        R[:, ng:ng+nx, ng:ng+ny] -= dFy[:, ng:ng+nx, :] / cfg.dy

        if cfg.powell_source and cfg.B_transverse > 0:
            S = self._compute_powell_source(U, W)
            R[:, ng:ng+nx, ng:ng+ny] += S[:, ng:ng+nx, ng:ng+ny]

        return R

    def step_ssprk3(self):
        dt = self.dt
        self._glm_ch_frozen = self.cfg.glm_ch
        U0 = self.U.copy()

        bflux = self._compute_boundary_energy_flux(U0)
        self._cumulative_boundary_energy += bflux * dt

        U1 = U0 + dt * self.compute_rhs(U0)
        U1 = self.enforce_scalar_bounds(U1)
        U1 = self.apply_bc(U1)

        U2 = 0.75*U0 + 0.25*(U1 + dt*self.compute_rhs(U1))
        U2 = self.enforce_scalar_bounds(U2)
        U2 = self.apply_bc(U2)

        self.U = (1.0/3.0)*U0 + (2.0/3.0)*(U2 + dt*self.compute_rhs(U2))
        self.U = self.enforce_scalar_bounds(self.U)
        self.U = self.apply_bc(self.U)

        if self._glm_ch_frozen > 0:
            decay = np.exp(-self.cfg.glm_alpha * self._glm_ch_frozen * dt / min(self.cfg.dx, self.cfg.dy))
            self.U[PSI] *= decay

    def compute_diagnostics(self):
        ng = self.ng; cfg = self.cfg
        W = cons_to_prim(self.U, cfg.gamma)

        rho = W[iRHO, ng:-ng, ng:-ng]
        p = W[iPR, ng:-ng, ng:-ng]
        vx = W[iVX, ng:-ng, ng:-ng]
        vy = W[iVY, ng:-ng, ng:-ng]
        Bx_f = W[iBX, ng:-ng, ng:-ng]
        By_f = W[iBY, ng:-ng, ng:-ng]
        Bz_f = W[iBZ, ng:-ng, ng:-ng]
        C = W[iCLR, ng:-ng, ng:-ng]

        x_arr = self.x[ng:-ng]; y_arr = self.y[ng:-ng]
        nx_int, ny_int = rho.shape

        C_clamped = np.clip(C, 0.0, 1.0)
        C_bar = np.mean(C_clamped, axis=1)
        integrand_mw = C_bar * (1.0 - C_bar)
        mw_integral = float(np.trapz(integrand_mw, x_arr))

        mixed = np.where((C_bar > 0.01) & (C_bar < 0.99))[0]
        mw_thresh = float(x_arr[mixed[-1]] - x_arr[mixed[0]]) if len(mixed) > 1 else 0.0

        C_bar_2d = C_bar[:, None]
        C_prime_sq = np.mean((C_clamped - C_bar_2d)**2, axis=1)
        denom_mix = np.maximum(C_bar * (1.0 - C_bar), 1e-14)
        if len(mixed) > 1:
            num = float(np.trapz(C_prime_sq[mixed], x_arr[mixed]))
            den = float(np.trapz(denom_mix[mixed], x_arr[mixed]))
            mixedness = float(np.clip(1.0 - num / max(den, 1e-14), 0.0, 1.0))
        else:
            mixedness = 0.0

        drho_dx = np.abs(np.gradient(C_clamped, cfg.dx, axis=0))
        interface_pos = np.full(ny_int, np.nan)
        for j in range(ny_int):
            grad_col = drho_dx[:, j]
            if np.max(grad_col) > 1e-6:
                ix_peak = np.argmax(grad_col)
                hw = min(10, ix_peak, nx_int - ix_peak - 1)
                if hw > 0:
                    sl = slice(ix_peak-hw, ix_peak+hw+1)
                    weights = grad_col[sl]
                    ws = np.sum(weights)
                    if ws > 1e-12:
                        interface_pos[j] = np.sum(x_arr[sl] * weights) / ws

        valid = ~np.isnan(interface_pos)
        n_valid = np.sum(valid)

        if n_valid > ny_int // 2:
            pos_valid = interface_pos[valid]; y_valid = y_arr[valid]
            pos_interp = np.interp(y_arr, y_valid, pos_valid) if n_valid < ny_int else pos_valid
            pos_fluct = pos_interp - np.mean(pos_interp)
            modes = np.fft.rfft(pos_fluct)
            mode_amps = 2.0 * np.abs(modes) / ny_int
            pert_amp = float(mode_amps[cfg.perturbation_mode]) if cfg.perturbation_mode < len(mode_amps) else 0.0
        else:
            mode_amps = np.zeros(ny_int // 2 + 1)
            pert_amp = 0.0

        dvydx = np.gradient(vy, cfg.dx, axis=0)
        dvxdy = np.gradient(vx, cfg.dy, axis=1)
        omega = dvydx - dvxdy
        enstrophy_global = float(np.mean(rho * omega**2))

        if n_valid > ny_int // 2:
            x_contact = float(np.mean(interface_pos[valid]))
            ix_lo = max(np.searchsorted(x_arr, x_contact - 0.5), 0)
            ix_hi = min(np.searchsorted(x_arr, x_contact + 0.5), nx_int)
            if ix_hi - ix_lo > 3:
                enstrophy_local = float(np.mean(rho[ix_lo:ix_hi,:] * omega[ix_lo:ix_hi,:]**2))
            else:
                enstrophy_local = enstrophy_global
        else:
            enstrophy_local = enstrophy_global

        v2 = vx**2 + vy**2
        B2 = Bx_f**2 + By_f**2 + Bz_f**2
        stag = float(np.max(p + 0.5*rho*v2 + 0.5*B2))

        divB_field = np.zeros_like(Bx_f)
        if Bx_f.shape[0] > 2 and Bx_f.shape[1] > 2:
            divB_field[1:-1, 1:-1] = (
                (0.5*(Bx_f[1:-1, 1:-1] + Bx_f[2:, 1:-1]) -
                 0.5*(Bx_f[:-2, 1:-1] + Bx_f[1:-1, 1:-1])) / cfg.dx +
                (0.5*(By_f[1:-1, 1:-1] + By_f[1:-1, 2:]) -
                 0.5*(By_f[1:-1, :-2] + By_f[1:-1, 1:-1])) / cfg.dy
            )
        divB_max = float(np.max(np.abs(divB_field)))
        divB_L2 = float(np.sqrt(np.mean(divB_field**2)))

        energy_total = float(np.sum(self.U[EN, ng:-ng, ng:-ng]) * cfg.dx * cfg.dy)

        self.diag_times.append(self.t)
        self.diag_mixing_width_integral.append(mw_integral)
        self.diag_mixing_width_thresh.append(mw_thresh)
        self.diag_mixedness.append(mixedness)
        self.diag_enstrophy.append(enstrophy_global)
        self.diag_enstrophy_local.append(enstrophy_local)
        self.diag_perturbation_amp.append(pert_amp)
        self.diag_mode_amps.append(mode_amps.copy())
        self.diag_stag_pressure.append(stag)
        self.diag_divB_max.append(divB_max)
        self.diag_divB_L2.append(divB_L2)
        self.diag_energy_total.append(energy_total)
        self.diag_boundary_flux_cumulative.append(self._cumulative_boundary_energy)

    def save_snapshot(self, label=None):
        ng = self.ng
        W = cons_to_prim(self.U, self.cfg.gamma)
        key = label or f"t={self.t:.4f}"
        self.snapshots[key] = {
            'rho': W[iRHO, ng:-ng, ng:-ng].copy(),
            'p': W[iPR, ng:-ng, ng:-ng].copy(),
            'vx': W[iVX, ng:-ng, ng:-ng].copy(),
            'vy': W[iVY, ng:-ng, ng:-ng].copy(),
            'Bx': W[iBX, ng:-ng, ng:-ng].copy(),
            'By': W[iBY, ng:-ng, ng:-ng].copy(),
            'Bz': W[iBZ, ng:-ng, ng:-ng].copy(),
            'C': W[iCLR, ng:-ng, ng:-ng].copy(),
            't': self.t,
            'x': self.x[ng:-ng].copy(),
            'y': self.y[ng:-ng].copy(),
        }

    def run(self):
        cfg = self.cfg; g = cfg.gamma
        print(f"
--- Running: B_y = {cfg.B_transverse} ---")
        sys.stdout.flush()
        t0 = time.time()

        self.compute_diagnostics()
        self.save_snapshot(label="initial")

        while self.t < cfg.t_end and self.step < cfg.max_steps:
            W = cons_to_prim(self.U, g)
            dt_cfl = self.compute_dt(W)
            self.dt = min(dt_cfl, cfg.t_end - self.t)
            if self.dt <= 1e-16: break
            self.step_ssprk3()
            self.t += self.dt
            self.step += 1

            if self.step % cfg.diag_interval == 0:
                self.compute_diagnostics()

            for st in cfg.snapshot_times:
                lbl = f"t~{st:.2f}"
                if abs(self.t - st) < 1.5*self.dt and lbl not in self.snapshots:
                    self.save_snapshot(label=lbl)

            if self.step % 200 == 0:
                mwi = self.diag_mixing_width_integral[-1] if self.diag_mixing_width_integral else 0
                theta = self.diag_mixedness[-1] if self.diag_mixedness else 0
                en = self.diag_enstrophy[-1] if self.diag_enstrophy else 0
                e_tot = self.diag_energy_total[-1] if self.diag_energy_total else 0
                divB = self.diag_divB_max[-1] if self.diag_divB_max else 0
                print(f"  Step {self.step:5d}  t={self.t:.4f}  dt={self.dt:.2e}  "
                      f"MW={mwi:.4f}  θ={theta:.3f}  enst={en:.1f}  "
                      f"divB={divB:.1f}  E={e_tot:.2f}")
                sys.stdout.flush()

        self.compute_diagnostics()
        self.save_snapshot(label="final")
        elapsed = time.time() - t0

        if len(self.diag_energy_total) >= 2:
            e0 = self.diag_energy_total[0]; ef = self.diag_energy_total[-1]
            bf = self._cumulative_boundary_energy
            de_raw = abs(ef-e0)/max(abs(e0),1e-14)*100
            de_corr = abs(ef-bf-e0)/max(abs(e0),1e-14)*100
            print(f"  Energy drift: raw={de_raw:.2f}%, corrected={de_corr:.2f}%")
            print(f"  Boundary flux: {bf:.2f}")

        if self.diag_divB_max:
            print(f"  Final max|divB|: {self.diag_divB_max[-1]:.2f}, "
                  f"L2|divB|: {self.diag_divB_L2[-1]:.4f}")

        print(f"  Done: {self.step} steps, {elapsed:.1f}s")
        sys.stdout.flush()


# ============================================================
# Verification Tests (Brio-Wu, Alfven wave, Contact)
# ============================================================
def brio_wu_test(nx=800, t_end=0.1, plot=True, output_dir="rmi_output_"):
    import matplotlib.pyplot as plt
    from mesh import build_grid
    cfg = Config(
        nx=nx, ny=4, x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.05,
        t_end=t_end, cfl=0.30, gamma=2.0, mach=1.0, B_transverse=0.0,
        interface_x=0.5, perturbation_amp=0.0, density_ratio=1.0,
        diag_interval=10000, snapshot_times=[t_end],
        powell_source=False, use_char_bc=False,
        bc_x_type="extrapolation", bc_y_type="periodic",
    )
    solver = MHDSolver(cfg)
    ng = solver.ng

    W = np.zeros((NVAR, solver.nx_tot, solver.ny_tot))
    x = solver.x
    left = x < 0.5

    W[iRHO] = np.where(left[:, None], 1.0, 0.125)
    W[iPR] = np.where(left[:, None], 1.0, 0.1)
    W[iBX] = 0.75
    W[iBY] = np.where(left[:, None], 1.0, -1.0)
    W[iCLR] = np.where(left[:, None], 1.0, 0.0)

    solver.U = prim_to_cons(W, cfg.gamma)
    solver.t = 0.0
    solver.step = 0

    t0 = time.time()
    while solver.t < cfg.t_end and solver.step < cfg.max_steps:
        W_c = cons_to_prim(solver.U, cfg.gamma)
        dt_cfl = solver.compute_dt(W_c)
        solver.dt = min(dt_cfl, cfg.t_end - solver.t)
        if solver.dt <= 1e-16: break
        solver.step_ssprk3()
        solver.t += solver.dt
        solver.step += 1
    elapsed = time.time() - t0
    print(f"  Brio-Wu: {solver.step} steps, {elapsed:.1f}s")

    W_final = cons_to_prim(solver.U, cfg.gamma)
    rho_1d = W_final[iRHO, ng:-ng, ng].copy()
    p_1d = W_final[iPR, ng:-ng, ng].copy()
    By_1d = W_final[iBY, ng:-ng, ng].copy()
    vx_1d = W_final[iVX, ng:-ng, ng].copy()
    x_1d = x[ng:-ng]

    rho_max = float(np.max(rho_1d))
    rho_min = float(np.min(rho_1d))

    check1 = 0.1 < rho_min < 0.2
    check2 = 0.9 < rho_max < 1.05
    n_levels = len(np.unique(np.round(rho_1d, 2)))
    check3 = n_levels > 10
    check4 = bool(np.any(By_1d[:-1] * By_1d[1:] < 0))
    check5 = rho_min < 0.18
    vx_range = float(np.max(vx_1d) - np.min(vx_1d))
    check6 = vx_range > 0.5

    passed = check1 and check2 and check3 and check4 and check5 and check6
    print(f"  rho range: [{rho_min:.4f}, {rho_max:.4f}]")
    print(f"  Distinct density levels: {n_levels}")
    print(f"  By sign change: {check4}, vx range: {vx_range:.3f}")
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, data, ylabel, title, color in [
            (axes[0,0], rho_1d, r'$ho$', 'Density', 'b'),
            (axes[0,1], p_1d, '$p$', 'Pressure', 'r'),
            (axes[1,0], vx_1d, '$v_x$', 'Velocity', 'g'),
            (axes[1,1], By_1d, '$B_y$', 'Transverse B', 'm'),
        ]:
            ax.plot(x_1d, data, f'{color}-', lw=1)
            ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3)
        axes[1,0].set_xlabel('$x$'); axes[1,1].set_xlabel('$x$')
        fig.suptitle(f'Brio-Wu Shock Tube, t={t_end}, nx={nx}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, 'brio_wu_test.png')
        fig.savefig(filepath, bbox_inches='tight', dpi=180)
        plt.close(fig)
        print(f"  ✓ Brio-Wu plot → {filepath}")
    return passed


def linear_wave_convergence_test(plot=True, output_dir="rmi_output_"):
    print("
--- Linear Alfven Wave Convergence Test  ---")
    sys.stdout.flush()

    resolutions = [32, 64, 128, 256]
    errors = []

    rho0 = 1.0; p0 = 0.1; Bx0 = 1.0; amp = 1e-6; gamma = 5.0/3.0; Lx = 1.0
    vA = Bx0 / np.sqrt(rho0)
    period = Lx / vA
    print(f"  Setup: rho0={rho0}, p0={p0}, Bx0={Bx0}, amp={amp:.0e}")
    print(f"  vA={vA:.4f}, period={period:.4f}, Lx={Lx}")

    for nx in resolutions:
        cfg = Config(
            nx=nx, ny=4, x_min=0.0, x_max=Lx, y_min=0.0, y_max=0.05,
            t_end=period, cfl=0.25, gamma=gamma, mach=1.0, B_transverse=0.0,
            interface_x=0.5, perturbation_amp=0.0, density_ratio=1.0,
            diag_interval=100000, snapshot_times=[period],
            powell_source=False, use_char_bc=False,
            bc_x_type="periodic", bc_y_type="periodic",
        )
        solver = MHDSolver(cfg)
        ng = solver.ng
        x = solver.x
        kx = 2.0 * np.pi / Lx

        W = np.zeros((NVAR, solver.nx_tot, solver.ny_tot))
        W[iRHO] = rho0
        W[iPR] = p0
        W[iBX] = Bx0
        W[iBY] = amp * np.sin(kx * x)[:, None] * np.ones(solver.ny_tot)[None, :]
        W[iVY] = -amp * np.sin(kx * x)[:, None] * np.ones(solver.ny_tot)[None, :] / np.sqrt(rho0)

        W_init = W.copy()
        solver.U = prim_to_cons(W, cfg.gamma)
        solver.t = 0.0; solver.step = 0

        while solver.t < cfg.t_end and solver.step < 100000:
            W_c = cons_to_prim(solver.U, cfg.gamma)
            dt_cfl = solver.compute_dt(W_c)
            solver.dt = min(dt_cfl, cfg.t_end - solver.t)
            if solver.dt <= 1e-16: break
            solver.step_ssprk3()
            solver.t += solver.dt
            solver.step += 1

        W_final = cons_to_prim(solver.U, cfg.gamma)
        By_init = W_init[iBY, ng:-ng, ng]
        By_final = W_final[iBY, ng:-ng, ng]
        L1_err = float(np.mean(np.abs(By_final - By_init)))
        errors.append(L1_err)
        print(f"  nx={nx:4d}: L1(By) = {L1_err:.2e}, steps={solver.step}, t_final={solver.t:.6f}")

    orders = []
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i-1] > 0:
            order = np.log(errors[i-1] / errors[i]) / np.log(resolutions[i] / resolutions[i-1])
            orders.append(order)
            print(f"  Order ({resolutions[i-1]}→{resolutions[i]}): {order:.2f}")

    mean_order = np.mean(orders) if orders else 0
    passed = mean_order >= 1.5
    print(f"  Mean convergence order: {mean_order:.2f} ({'PASS ✓' if passed else 'FAIL ✗'})")

    if plot and len(errors) > 1:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(resolutions, errors, 'bo-', lw=2, ms=8, label=f'Measured (order={mean_order:.2f})')
        ref_x = np.array([resolutions[0], resolutions[-1]], dtype=float)
        scale = errors[0] * (resolutions[0])**2
        ax.loglog(ref_x, scale / ref_x**2, 'k--', lw=1, alpha=0.5, label='2nd order')
        scale1 = errors[0] * resolutions[0]
        ax.loglog(ref_x, scale1 / ref_x, 'k:', lw=1, alpha=0.5, label='1st order')
        ax.set_xlabel('Resolution $N_x$'); ax.set_ylabel('$L_1$ error in $B_y$')
        ax.set_title('Alfven Wave Convergence Test', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, 'convergence_alfven.png')
        fig.savefig(filepath, bbox_inches='tight', dpi=180)
        plt.close(fig)
        print(f"  ✓ Convergence plot → {filepath}")
    return passed


def contact_discontinuity_test(plot=True, output_dir="rmi_output_"):
    print("
--- Contact Discontinuity Test ---")
    sys.stdout.flush()

    cfg = Config(
        nx=400, ny=4, x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.01,
        t_end=0.2, cfl=0.30, gamma=5.0/3.0, mach=1.0, B_transverse=0.0,
        interface_x=0.5, perturbation_amp=0.0, density_ratio=1.0,
        diag_interval=100000, snapshot_times=[0.2],
        powell_source=False, use_char_bc=False,
        bc_x_type="extrapolation", bc_y_type="periodic",
    )
    solver = MHDSolver(cfg)
    ng = solver.ng

    W = np.zeros((NVAR, solver.nx_tot, solver.ny_tot))
    x = solver.x
    W[iRHO] = np.where(x[:, None] < 0.5, 1.0, 3.0)
    W[iPR] = 1.0
    W[iVX] = 1.0
    W[iBX] = 0.5
    W[iCLR] = np.where(x[:, None] < 0.5, 0.0, 1.0)

    solver.U = prim_to_cons(W, cfg.gamma)
    solver.t = 0.0; solver.step = 0

    while solver.t < cfg.t_end and solver.step < cfg.max_steps:
        W_c = cons_to_prim(solver.U, cfg.gamma)
        dt_cfl = solver.compute_dt(W_c)
        solver.dt = min(dt_cfl, cfg.t_end - solver.t)
        if solver.dt <= 1e-16: break
        solver.step_ssprk3()
        solver.t += solver.dt
        solver.step += 1

    W_final = cons_to_prim(solver.U, cfg.gamma)
    rho_1d = W_final[iRHO, ng:-ng, ng]
    p_1d = W_final[iPR, ng:-ng, ng]
    vx_1d = W_final[iVX, ng:-ng, ng]
    x_1d = x[ng:-ng]

    p_variation = float(np.max(p_1d) - np.min(p_1d)) / float(np.mean(p_1d))
    v_variation = float(np.max(vx_1d) - np.min(vx_1d))
    rho_min = float(np.min(rho_1d)); rho_max = float(np.max(rho_1d))

    check1 = p_variation < 0.05
    check2 = v_variation < 0.1
    check3 = rho_min > 0.9 and rho_max < 3.1
    passed = check1 and check2 and check3
    print(f"  Pressure variation: {p_variation*100:.2f}% (need <5%)")
    print(f"  Velocity variation: {v_variation:.4f} (need <0.1)")
    print(f"  Density range: [{rho_min:.3f}, {rho_max:.3f}]")
    print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(x_1d, rho_1d, 'b-', lw=1); axes[0].set_ylabel(r'$ho$')
        axes[0].set_title('Density'); axes[0].axvline(0.7, color='r', ls='--', alpha=0.5)
        axes[1].plot(x_1d, p_1d, 'r-', lw=1); axes[1].set_ylabel('$p$')
        axes[1].set_title('Pressure')
        axes[2].plot(x_1d, vx_1d, 'g-', lw=1); axes[2].set_ylabel('$v_x$')
        axes[2].set_title('Velocity')
        for ax in axes: ax.grid(True, alpha=0.3); ax.set_xlabel('$x$')
        fig.suptitle('Contact Discontinuity Test (t=0.2)', fontweight='bold')
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, 'contact_test.png')
        fig.savefig(filepath, bbox_inches='tight', dpi=180)
        plt.close(fig)
        print(f"  ✓ Contact test plot → {filepath}")
    return passed


# ============================================================
# Convergence study
# ============================================================
def run_convergence_study(base_params, By_value=0.0, resolutions=None, plot=True, output_dir="rmi_output_"):
    if resolutions is None:
        resolutions = [(100, 50), (200, 100), (400, 200)]

    print(f"
{'='*64}")
    print(f"  CONVERGENCE STUDY: By={By_value}")
    print(f"  Resolutions: {resolutions}")
    print(f"{'='*64}")
    sys.stdout.flush()

    results = {'resolutions': resolutions, 'dx': [], 'solvers': []}

    for nx, ny in resolutions:
        print(f"
--- Resolution {nx}x{ny} ---")
        params = base_params.copy()
        params['nx'] = nx
        params['ny'] = ny
        params['B_transverse'] = By_value
        params['diag_interval'] = max(10, 20 * 200 // nx)

        cfg = Config(**params)
        solver = MHDSolver(cfg)
        solver.initialize()
        solver.run()
        results['solvers'].append(solver)
        results['dx'].append(cfg.dx)

    print(f"
  Convergence Results:")
    print(f"  {'nx':>6s} {'ny':>6s} {'dx':>10s} {'MW_int':>10s} {'MW_thr':>10s} {'Peak_enst':>12s}")
    print(f"  {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    mw_ints = []
    mw_thrs = []
    for (nx, ny), solver in zip(resolutions, results['solvers']):
        t = np.array(solver.diag_times)
        mask = t > 0.03
        mwi_arr = np.array(solver.diag_mixing_width_integral)[mask]
        mwt_arr = np.array(solver.diag_mixing_width_thresh)[mask]
        enst_arr = np.array(solver.diag_enstrophy_local)[mask]

        peak_mwi = float(np.max(mwi_arr)) if len(mwi_arr) > 0 else 0
        peak_mwt = float(np.max(mwt_arr)) if len(mwt_arr) > 0 else 0
        peak_enst = float(np.max(enst_arr)) if len(enst_arr) > 0 else 0
        mw_ints.append(peak_mwi)
        mw_thrs.append(peak_mwt)

        dx = solver.cfg.dx
        print(f"  {nx:6d} {ny:6d} {dx:10.5f} {peak_mwi:10.5f} {peak_mwt:10.5f} {peak_enst:12.1f}")

    if len(resolutions) >= 3:
        dx_arr = np.array(results['dx'])
        mwi_arr_conv = np.array(mw_ints)
        for i in range(1, len(mwi_arr_conv)):
            if mwi_arr_conv[i] > 0 and mwi_arr_conv[i-1] > 0:
                diff = abs(mwi_arr_conv[i] - mwi_arr_conv[i-1])
                if i >= 2:
                    diff_prev = abs(mwi_arr_conv[i-1] - mwi_arr_conv[i-2])
                    if diff > 1e-14 and diff_prev > 1e-14:
                        order = np.log(diff_prev / diff) / np.log(dx_arr[i-1] / dx_arr[i])
                        print(f"  Estimated convergence order (MW_int): {order:.2f}")

    results['mw_ints'] = mw_ints
    results['mw_thrs'] = mw_thrs

    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors_conv = ['C0', 'C1', 'C2', 'C3']

        ax = axes[0]
        for i, ((nx, ny), solver) in enumerate(zip(resolutions, results['solvers'])):
            t = np.array(solver.diag_times)
            mw = np.array(solver.diag_mixing_width_integral)
            ax.plot(t, mw, color=colors_conv[i], lw=2, label=f'{nx}×{ny}')
        ax.set_xlabel('Time $t$'); ax.set_ylabel('Integral MW')
        ax.set_title('Mixing Width Convergence', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        ax = axes[1]
        for i, ((nx, ny), solver) in enumerate(zip(resolutions, results['solvers'])):
            t = np.array(solver.diag_times)
            en = np.array(solver.diag_enstrophy_local)
            ax.plot(t, en, color=colors_conv[i], lw=2, label=f'{nx}×{ny}')
        ax.set_xlabel('Time $t$'); ax.set_ylabel('Local Enstrophy')
        ax.set_title('Enstrophy Convergence', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        ax = axes[2]
        for i, ((nx, ny), solver) in enumerate(zip(resolutions, results['solvers'])):
            snap = solver.snapshots.get('final', list(solver.snapshots.values())[-1])
            rho = snap['rho']
            j_mid = rho.shape[1] // 2
            ax.plot(snap['x'], rho[:, j_mid], color=colors_conv[i], lw=1.5, label=f'{nx}×{ny}')
        ax.set_xlabel('$x$'); ax.set_ylabel(r'$ho$ at $y=L_y/2$')
        ax.set_title('Midline Density Profile', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

        By_str = f'By={By_value}' if By_value > 0 else 'Hydro'
        fig.suptitle(f'Grid Convergence Study ({By_str})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        fname = f'convergence_{By_str.replace("=","").replace(".","p")}.png'
        filepath = os.path.join(output_dir, fname)
        fig.savefig(filepath, bbox_inches='tight', dpi=180)
        plt.close(fig)
        print(f"  ✓ Convergence plot → {filepath}")

    sys.stdout.flush()
    return results
