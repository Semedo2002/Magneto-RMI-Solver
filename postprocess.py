"""
postprocess.py
PostProcessor class for generating all plots from simulation results.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from config import iRHO, iPR, iVX, iVY, iBX, iBY, iBZ, iCLR, NVAR
from solver import smooth, richtmyer_linear_theory

OUTPUT_DIR = "rmi_output_"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_figure(fig, filename):
    """Save figure reliably and verify the file exists."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=180)
    plt.close(fig)
    if os.path.exists(filepath):
        size_kb = os.path.getsize(filepath) / 1024
        return True, filepath, size_kb
    return False, filepath, 0


class PostProcessor:
    """Generate all analysis plots from simulation results."""

    def __init__(self, solvers_dict):
        self.solvers = solvers_dict
        self._colors = ['C3', 'C2', 'C0', 'C4', 'C5']
        self._styles = ['-', '--', '-.', ':', '-']
        self._saved_files = []

    @staticmethod
    def _final(s):
        return s.snapshots.get('final', list(s.snapshots.values())[-1])

    def _save(self, fig, filename):
        ok, path, sz = save_figure(fig, filename)
        self._saved_files.append((filename, ok, sz))
        return ok, path, sz

    def _get_smoothing(self):
        for s in self.solvers.values():
            return s.cfg.enable_smoothing
        return False

    def plot_density_comparison(self):
        n = len(self.solvers)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, self.solvers.items()):
            snap = self._final(solver)
            rho = snap['rho']
            im = ax.pcolormesh(snap['x'], snap['y'], rho.T,
                               cmap='inferno', shading='auto',
                               vmin=rho.min(), vmax=np.percentile(rho, 98))
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}\n$t={snap['t']:.3f}$", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$ho$')
        axes[0].set_ylabel('$y$')
        fig.suptitle(r'Magnetized RMI — Mach 10 Shock', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'rmi_density.png')
        print(f"  ✓ Density comparison")

    def plot_passive_scalar(self):
        n = len(self.solvers)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, self.solvers.items()):
            snap = self._final(solver)
            C = np.clip(snap['C'], 0, 1)
            im = ax.pcolormesh(snap['x'], snap['y'], C.T, cmap='coolwarm',
                               shading='auto', vmin=0, vmax=1)
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}: Color $C$\n$t={snap['t']:.3f}$", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label='$C$')
        axes[0].set_ylabel('$y$')
        fig.suptitle('Passive Scalar (Mixing Tracer)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'rmi_passive_scalar.png')
        print(f"  ✓ Passive scalar")

    def plot_schlieren(self):
        n = len(self.solvers)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, self.solvers.items()):
            snap = self._final(solver)
            rho = snap['rho']
            dx, dy = solver.cfg.dx, solver.cfg.dy
            grad_rho = np.sqrt(np.gradient(rho, dx, axis=0)**2 + np.gradient(rho, dy, axis=1)**2)
            schlieren = np.log10(grad_rho / np.maximum(rho, 1e-10) + 1e-10)
            vmin_s = np.percentile(schlieren, 3)
            vmax_s = np.percentile(schlieren, 99)
            im = ax.pcolormesh(snap['x'], snap['y'], schlieren.T, cmap='gray_r',
                               shading='auto', vmin=vmin_s, vmax=vmax_s)
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}: Schlieren\n$t={snap['t']:.3f}$", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$\log_{10}(|
ablaho|/ho)$')
        axes[0].set_ylabel('$y$')
        fig.suptitle('Numerical Schlieren', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'rmi_schlieren.png')
        print(f"  ✓ Schlieren")

    def plot_vorticity_comparison(self):
        n = len(self.solvers)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        omegas, snaps = [], []
        for label, solver in self.solvers.items():
            snap = self._final(solver); snaps.append(snap)
            dvydx = np.gradient(snap['vy'], solver.cfg.dx, axis=0)
            dvxdy = np.gradient(snap['vx'], solver.cfg.dy, axis=1)
            omegas.append(dvydx - dvxdy)
        vmax_om = max(np.percentile(np.abs(om), 99) for om in omegas)
        vmax_om = max(vmax_om, 1e-10)
        for ax, (label, _), snap, omega in zip(axes, self.solvers.items(), snaps, omegas):
            im = ax.pcolormesh(snap['x'], snap['y'], omega.T, cmap='RdBu_r',
                               shading='auto', vmin=-vmax_om, vmax=vmax_om)
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}: $\\omega_z$\n$t={snap['t']:.3f}$", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$\omega_z$')
        axes[0].set_ylabel('$y$')
        fig.suptitle('Vorticity Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'rmi_vorticity.png')
        print(f"  ✓ Vorticity")

    def plot_magnetic_pressure_with_fieldlines(self):
        mhd = {k: v for k, v in self.solvers.items() if v.cfg.B_transverse > 0}
        if not mhd:
            print("  ⊘ No MHD cases for magnetic pressure plot"); return
        n = len(mhd)
        fig, axes = plt.subplots(1, n, figsize=(5.5*n, 5), sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (label, solver) in zip(axes, mhd.items()):
            snap = self._final(solver)
            Pmag = 0.5*(snap['Bx']**2 + snap['By']**2 + snap['Bz']**2)
            im = ax.pcolormesh(snap['x'], snap['y'], Pmag.T, cmap='plasma', shading='auto')
            try:
                ax.streamplot(snap['x'], snap['y'], snap['Bx'].T, snap['By'].T,
                              color='white', linewidth=0.5, density=1.2, arrowsize=0.5)
            except Exception:
                pass
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"{label}: $P_B$ + field lines\n$t={snap['t']:.3f}$", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$P_B=B^2/2$')
        axes[0].set_ylabel('$y$')
        fig.suptitle('Magnetic Pressure & Field Lines', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'rmi_mag_pressure.png')
        print(f"  ✓ Magnetic pressure")

    def plot_diagnostics(self):
        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        colors, styles = self._colors, self._styles
        sm_en = self._get_smoothing()
        ns = 5
        pairs = list(self.solvers.items())

        ax = axes[0, 0]
        for i, (label, s) in enumerate(pairs):
            t, mw = np.array(s.diag_times), np.array(s.diag_mixing_width_integral)
            ax.plot(t, smooth(mw, ns, sm_en), color=colors[i%5], ls=styles[i%5], lw=2, label=label)
            ax.plot(t, mw, color=colors[i%5], lw=0.4, alpha=0.15)
        ax.set_xlabel('$t$'); ax.set_ylabel(r'$W = \int \langle Cangle(1-\langle Cangle)\,dx$')
        ax.set_title('Integral Mixing Width', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        ax = axes[0, 1]
        for i, (label, s) in enumerate(pairs):
            t, mw = np.array(s.diag_times), np.array(s.diag_mixing_width_thresh)
            ax.plot(t, smooth(mw, ns, sm_en), color=colors[i%5], ls=styles[i%5], lw=2, label=label)
            ax.plot(t, mw, color=colors[i%5], lw=0.4, alpha=0.15)
        ax.set_xlabel('$t$'); ax.set_ylabel('Threshold Width')
        ax.set_title('Threshold MW (1%–99%)', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        ax = axes[0, 2]
        for i, (label, s) in enumerate(pairs):
            t, theta = np.array(s.diag_times), np.array(s.diag_mixedness)
            ax.plot(t, smooth(theta, ns, sm_en), color=colors[i%5], ls=styles[i%5], lw=2, label=label)
            ax.plot(t, theta, color=colors[i%5], lw=0.4, alpha=0.15)
        ax.set_xlabel('$t$'); ax.set_ylabel(r'$	heta$')
        ax.set_title(r'Mixedness $	heta$', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(0, 1)

        ax = axes[0, 3]
        for i, (label, s) in enumerate(pairs):
            t, amp = np.array(s.diag_times), np.array(s.diag_perturbation_amp)
            ax.plot(t, smooth(amp, ns, sm_en), color=colors[i%5], ls=styles[i%5], lw=2, label=label)
            ax.plot(t, amp, color=colors[i%5], lw=0.4, alpha=0.15)
            if s._rh_results is not None:
                cfg = s.cfg
                Ly = cfg.y_max - cfg.y_min
                k_pert = 2*np.pi*cfg.perturbation_mode / Ly
                t_theory = np.linspace(0, cfg.t_end, 500)
                a_lin, t_sh, info = richtmyer_linear_theory(
                    t_theory, cfg.gamma, cfg.mach, 1.0,
                    cfg.density_ratio, cfg.perturbation_amp, k_pert,
                    By=cfg.B_transverse)
                ax.plot(t_theory, a_lin, color=colors[i%5], ls=':', lw=1.5, alpha=0.7)
        ax.set_xlabel('$t$'); ax.set_ylabel(r'Mode amplitude $a_k$')
        ax.set_title(f'Mode {pairs[0][1].cfg.perturbation_mode} Amp + Linear Theory', fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        ax = axes[1, 0]
        for i, (label, s) in enumerate(pairs):
            ax.plot(s.diag_times, s.diag_enstrophy_local, color=colors[i%5],
                    ls=styles[i%5], lw=2, label=label)
        ax.set_xlabel('$t$'); ax.set_ylabel(r'$\langleho\omega_z^2angle_\mathrm{local}$')
        ax.set_title('Local Enstrophy (interface)', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        ax = axes[1, 1]
        for i, (label, s) in enumerate(pairs):
            ax.plot(s.diag_times, s.diag_stag_pressure, color=colors[i%5],
                    ls=styles[i%5], lw=2, label=label)
        ax.set_xlabel('$t$'); ax.set_ylabel('Peak Stag. Pressure')
        ax.set_title('Stagnation Pressure', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)

        ax = axes[1, 2]
        has_mhd = False
        for i, (label, s) in enumerate(pairs):
            if s.cfg.B_transverse > 0:
                ax.plot(s.diag_times, s.diag_divB_max, color=colors[i%5],
                        ls=styles[i%5], lw=2, label=f'{label} max')
                ax.plot(s.diag_times, s.diag_divB_L2, color=colors[i%5],
                        ls=':', lw=1.5, alpha=0.7, label=f'{label} L2')
                has_mhd = True
        if has_mhd:
            ax.set_xlabel('$t$'); ax.set_ylabel(r'$|
abla\cdot\mathbf{B}|$')
            ax.set_title(r'$
abla\cdot\mathbf{B}$ Control (max & L2)', fontweight='bold')
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        else:
            ax.set_visible(False)

        ax = axes[1, 3]
        for i, (label, s) in enumerate(pairs):
            t = np.array(s.diag_times)
            e = np.array(s.diag_energy_total)
            bf = np.array(s.diag_boundary_flux_cumulative)
            if len(e) > 0 and abs(e[0]) > 1e-14:
                e_rel_raw = (e - e[0]) / abs(e[0]) * 100
                ax.plot(t, e_rel_raw, color=colors[i%5], ls=styles[i%5], lw=1, alpha=0.3)
                e_corrected = (e - bf - e[0]) / abs(e[0]) * 100
                ax.plot(t, e_corrected, color=colors[i%5], ls=styles[i%5],
                        lw=2, label=f"{label} (corrected)")
        ax.set_xlabel('$t$'); ax.set_ylabel(r'$\Delta E / E_0$ (%)')
        ax.set_title('Energy Conservation', fontweight='bold')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)

        plt.tight_layout()
        self._save(fig, 'rmi_diagnostics.png')
        print(f"  ✓ Diagnostics")

    def plot_evolution(self):
        n_cases = len(self.solvers)
        n_cols = 4
        fig, axes = plt.subplots(n_cases, n_cols, figsize=(4*n_cols, 3*n_cases),
                                 sharex=True, sharey=True, squeeze=False)
        for row, (label, solver) in enumerate(self.solvers.items()):
            keys = sorted(solver.snapshots.keys(), key=lambda k: solver.snapshots[k]['t'])
            if len(keys) > n_cols:
                idx = np.linspace(0, len(keys)-1, n_cols, dtype=int)
                keys = [keys[i] for i in idx]
            for col in range(min(n_cols, len(keys))):
                ax = axes[row, col]
                snap = solver.snapshots[keys[col]]
                rho = snap['rho']
                ax.pcolormesh(snap['x'], snap['y'], rho.T, cmap='inferno',
                              shading='auto', vmin=0.5, vmax=rho.max()*0.95)
                ax.set_title(f"$t={snap['t']:.3f}$", fontsize=10)
                ax.set_aspect('equal')
                if col == 0: ax.set_ylabel(f"{label}\n$y$")
                if row == n_cases-1: ax.set_xlabel('$x$')
        fig.suptitle('Density Evolution', fontsize=13, fontweight='bold')
        plt.tight_layout()
        self._save(fig, 'rmi_evolution.png')
        print(f"  ✓ Evolution strips")

    def plot_hero_enstrophy(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (label, s) in enumerate(self.solvers.items()):
            ax.plot(s.diag_times, s.diag_enstrophy_local, color=self._colors[i%5],
                    ls=self._styles[i%5], lw=2.5, label=label)
        ax.set_xlabel('Time $t$', fontsize=13)
        ax.set_ylabel(r'Local Enstrophy $\langleho\omega_z^2angle$', fontsize=13)
        ax.set_title('Enstrophy: Vorticity Suppression by Magnetic Field',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        plt.tight_layout()
        self._save(fig, 'rmi_enstrophy_hero.png')
        print(f"  ✓ Hero enstrophy")

    def plot_hero_density(self):
        keys = list(self.solvers.keys())
        if len(keys) < 2: return
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, key in zip(axes, [keys[0], keys[-1]]):
            snap = self._final(self.solvers[key])
            rho = snap['rho']
            im = ax.pcolormesh(snap['x'], snap['y'], rho.T, cmap='inferno',
                               shading='auto', vmin=rho.min(), vmax=np.percentile(rho, 98))
            ax.set_xlabel('$x$'); ax.set_aspect('equal')
            ax.set_title(f"$\\mathbf{{{key}}}$\n$t={snap['t']:.3f}$", fontweight='bold')
            fig.colorbar(im, ax=ax, shrink=0.82, label=r'$ho$')
        axes[0].set_ylabel('$y$')
        fig.suptitle(r'Magnetized RMI — Mach 10 Shock', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        self._save(fig, 'rmi_hero_density.png')
        print(f"  ✓ Hero density")

    def plot_interface_shape(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (label, solver) in enumerate(self.solvers.items()):
            snap = self._final(solver)
            C = np.clip(snap['C'], 0, 1)
            ax.contour(snap['x'], snap['y'], C.T, levels=[0.5],
                       colors=[self._colors[i%5]], linewidths=2)
            ax.plot([], [], color=self._colors[i%5], lw=2, label=label)
        ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
        ax.set_title('Interface Shape at Final Time ($C=0.5$)', fontweight='bold')
        ax.legend(); ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        plt.tight_layout()
        self._save(fig, 'rmi_interface_shape.png')
        print(f"  ✓ Interface shape")

    def plot_spectral_modes(self):
        keys = list(self.solvers.keys())
        hydro_key = keys[0]
        has_mhd = any(s.cfg.B_transverse > 0 for s in self.solvers.values())

        ncols = 2 if has_mhd else 1
        fig, axes = plt.subplots(1, ncols, figsize=(7*ncols, 5))
        if ncols == 1:
            axes = [axes]
        else:
            axes = list(axes)

        ax = axes[0]
        for i, (label, s) in enumerate(self.solvers.items()):
            if len(s.diag_mode_amps) > 0:
                amps = s.diag_mode_amps[-1]
                n_modes = min(len(amps), 20)
                ax.semilogy(range(n_modes), amps[:n_modes], color=self._colors[i%5],
                            ls=self._styles[i%5], lw=2, marker='o', ms=4, label=label)
        ax.set_xlabel('Mode number $k$'); ax.set_ylabel(r'Amplitude $|\hat{\eta}_k|$')
        ax.set_title('Interface Spectral Modes', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)

        if has_mhd and len(axes) > 1:
            ax = axes[1]
            hydro_amps = self.solvers[hydro_key].diag_mode_amps[-1] if self.solvers[hydro_key].diag_mode_amps else None
            if hydro_amps is not None:
                for i, (label, s) in enumerate(self.solvers.items()):
                    if s.cfg.B_transverse > 0 and len(s.diag_mode_amps) > 0:
                        mhd_amps = s.diag_mode_amps[-1]
                        n_modes = min(len(mhd_amps), len(hydro_amps), 20)
                        ratio = np.zeros(n_modes)
                        for k in range(n_modes):
                            if hydro_amps[k] > 1e-16:
                                ratio[k] = mhd_amps[k] / hydro_amps[k]
                        ax.plot(range(n_modes), ratio, color=self._colors[i%5],
                                ls=self._styles[i%5], lw=2, marker='s', ms=4, label=label)
                ax.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.5)
                ax.set_xlabel('Mode number $k$')
                ax.set_ylabel(r'$|\hat{\eta}_k^{MHD}| / |\hat{\eta}_k^{hydro}|$')
                ax.set_title('MHD/Hydro Spectral Ratio', fontweight='bold')
                ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)
                ax.set_ylim(bottom=0)

        plt.tight_layout()
        self._save(fig, 'rmi_spectral_modes.png')
        print(f"  ✓ Spectral modes")

    def plot_summary_bars(self):
        t_skip = 0.03
        labels_list, mwi_peaks, mwt_peaks, theta_means, enst_means = [], [], [], [], []
        sm_en = self._get_smoothing()
        for label, s in self.solvers.items():
            t = np.array(s.diag_times); mask = t > t_skip
            mwi = smooth(np.array(s.diag_mixing_width_integral)[mask], 7, sm_en)
            mwt = smooth(np.array(s.diag_mixing_width_thresh)[mask], 7, sm_en)
            theta = np.array(s.diag_mixedness)[mask]
            en = np.array(s.diag_enstrophy_local)[mask]
            labels_list.append(label)
            mwi_peaks.append(float(np.max(mwi)) if len(mwi) > 0 else 0)
            mwt_peaks.append(float(np.max(mwt)) if len(mwt) > 0 else 0)
            theta_means.append(float(np.mean(theta)) if len(theta) > 0 else 0)
            enst_means.append(float(np.mean(en)) if len(en) > 0 else 0)

        x_pos = np.arange(len(labels_list))
        bar_colors = self._colors[:len(labels_list)]
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 4.5))

        for ax, vals, ylabel, title in [
            (ax1, mwi_peaks, r'Peak $\int\langle Cangle(1-\langle Cangle)dx$', 'Integral MW'),
            (ax2, mwt_peaks, 'Peak Threshold Width', 'Threshold MW'),
            (ax3, theta_means, r'Mean $	heta$', 'Mixedness'),
            (ax4, enst_means, 'Mean Local Enstrophy', 'Enstrophy'),
        ]:
            bars = ax.bar(x_pos, vals, color=bar_colors, edgecolor='k')
            ax.set_xticks(x_pos); ax.set_xticklabels(labels_list, fontsize=8)
            ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            fmt = '.4f' if max(vals) < 1 else '.0f'
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                        f"{val:{fmt}}", ha='center', va='bottom', fontsize=9)
        ax3.set_ylim(0, 1)
        plt.tight_layout()
        self._save(fig, 'rmi_summary.png')
        print(f"  ✓ Summary bars")

    def plot_mixing_width_hero(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors, styles = self._colors, self._styles
        sm_en = self._get_smoothing()
        ns = 7
        pairs = list(self.solvers.items())

        ax = axes[0]
        for i, (label, s) in enumerate(pairs):
            t = np.array(s.diag_times)
            mw = smooth(np.array(s.diag_mixing_width_integral), ns, sm_en)
            ax.plot(t, mw, color=colors[i%5], ls=styles[i%5], lw=2.5, label=label)
        ax.set_xlabel('Time $t$', fontsize=13)
        ax.set_ylabel(r'$W = \int \langle Cangle(1-\langle Cangle)\,dx$', fontsize=12)
        ax.set_title('Integral Mixing Width', fontweight='bold', fontsize=14)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        ax = axes[1]
        for i, (label, s) in enumerate(pairs):
            t = np.array(s.diag_times)
            mw = smooth(np.array(s.diag_mixing_width_thresh), ns, sm_en)
            ax.plot(t, mw, color=colors[i%5], ls=styles[i%5], lw=2.5, label=label)
        ax.set_xlabel('Time $t$', fontsize=13)
        ax.set_ylabel('Threshold Width (1%–99%)', fontsize=12)
        ax.set_title('Threshold Mixing Width', fontweight='bold', fontsize=14)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        plt.tight_layout()
        self._save(fig, 'rmi_mixing_width_hero.png')
        print(f"  ✓ Mixing width hero")

    def plot_linear_theory_comparison(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors, styles = self._colors, self._styles

        ax = axes[0]
        for i, (label, s) in enumerate(self.solvers.items()):
            t = np.array(s.diag_times)
            amp = np.array(s.diag_perturbation_amp)
            ax.plot(t, amp, color=colors[i%5], ls=styles[i%5], lw=2.5, label=f'{label} (sim)')

            if s._rh_results is not None:
                cfg = s.cfg
                Ly = cfg.y_max - cfg.y_min
                k_pert = 2*np.pi*cfg.perturbation_mode / Ly
                t_theory = np.linspace(0, cfg.t_end, 500)
                a_lin, t_sh, info = richtmyer_linear_theory(
                    t_theory, cfg.gamma, cfg.mach, 1.0,
                    cfg.density_ratio, cfg.perturbation_amp, k_pert,
                    By=cfg.B_transverse)
                ax.plot(t_theory, a_lin, color=colors[i%5], ls=':', lw=1.5, alpha=0.7,
                        label=f'{label} (linear)')

        ax.set_xlabel('Time $t$', fontsize=13)
        ax.set_ylabel(r'Perturbation Amplitude $a_k$', fontsize=12)
        ax.set_title('Simulation vs Richtmyer Linear Theory', fontweight='bold', fontsize=13)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        ax = axes[1]
        keys = list(self.solvers.keys())
        if len(keys) >= 2:
            hydro_s = self.solvers[keys[0]]
            mhd_s = self.solvers[keys[-1]]
            t_h = np.array(hydro_s.diag_times)
            t_m = np.array(mhd_s.diag_times)
            amp_h = np.array(hydro_s.diag_perturbation_amp)
            amp_m = np.array(mhd_s.diag_perturbation_amp)

            vx_h = hydro_s._rh_results['vx2'] if hydro_s._rh_results else 1.0
            vx_m = mhd_s._rh_results['vx2'] if mhd_s._rh_results else 1.0

            ax.plot(t_h * vx_h, amp_h / hydro_s.cfg.perturbation_amp,
                    color=colors[0], ls=styles[0], lw=2.5, label=f'{keys[0]} (v_post·t)')
            ax.plot(t_m * vx_m, amp_m / mhd_s.cfg.perturbation_amp,
                    color=colors[2], ls=styles[2], lw=2.5, label=f'{keys[-1]} (v_post·t)')

        ax.set_xlabel(r'Normalized time $v_{post} \cdot t$', fontsize=12)
        ax.set_ylabel(r'$a_k / a_0$', fontsize=12)
        ax.set_title('Velocity-Normalized Comparison', fontweight='bold', fontsize=13)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        plt.tight_layout()
        self._save(fig, 'rmi_linear_theory.png')
        print(f"  ✓ Linear theory comparison")

    def plot_interface_perturbation(self):
        keys = list(self.solvers.keys())
        if len(keys) < 2: return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors, styles = self._colors, self._styles
        hydro_key, mhd_key = keys[0], keys[-1]

        for ax_idx, (metric_name, metric_key) in enumerate([
            ('Mixing Width (peak-to-trough)', 'diag_mixing_width_thresh'),
            ('Perturbation Amplitude', 'diag_perturbation_amp'),
        ]):
            ax = axes[ax_idx]
            for i, key in enumerate([hydro_key, mhd_key]):
                s = self.solvers[key]
                t = np.array(s.diag_times)
                val = np.array(getattr(s, metric_key))
                ax.plot(t, val, color=colors[i%5], ls=styles[i%5], lw=2.5, label=key)
            ax.set_xlabel('Time $t$', fontsize=13)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f'Interface {metric_name}', fontweight='bold', fontsize=13)
            ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        plt.tight_layout()
        self._save(fig, 'rmi_interface_perturbation.png')
        print(f"  ✓ Interface perturbation")

    def plot_stagnation_pressure(self):
        keys = list(self.solvers.keys())
        if len(keys) < 2: return
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, key in enumerate([keys[0], keys[-1]]):
            s = self.solvers[key]
            ax.plot(s.diag_times, s.diag_stag_pressure, color=self._colors[i%5],
                    ls=self._styles[i%5], lw=2.5, label=key)
        ax.set_xlabel('Time $t$', fontsize=13)
        ax.set_ylabel('Peak Stagnation Pressure', fontsize=12)
        ax.set_title('Stagnation Pressure', fontweight='bold', fontsize=14)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.3); ax.set_xlim(left=0)
        plt.tight_layout()
        self._save(fig, 'rmi_stagnation_pressure.png')
        print(f"  ✓ Stagnation pressure")

    def plot_divB_comparison(self):
        mhd_cases = {k: v for k, v in self.solvers.items() if v.cfg.B_transverse > 0}
        if not mhd_cases: return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors, styles = self._colors, self._styles

        ax = axes[0]
        for i, (label, s) in enumerate(mhd_cases.items()):
            ax.plot(s.diag_times, s.diag_divB_max, color=colors[i%5],
                    ls=styles[i%5], lw=2, label=label)
        ax.set_xlabel('Time $t$'); ax.set_ylabel(r'max $|
abla\cdot\mathbf{B}|$')
        ax.set_title(r'$
abla\cdot\mathbf{B}$ Control (max norm)', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        ax = axes[1]
        for i, (label, s) in enumerate(mhd_cases.items()):
            ax.plot(s.diag_times, s.diag_divB_L2, color=colors[i%5],
                    ls=styles[i%5], lw=2, label=label)
        ax.set_xlabel('Time $t$'); ax.set_ylabel(r'$L_2$ norm $|
abla\cdot\mathbf{B}|$')
        ax.set_title(r'$
abla\cdot\mathbf{B}$ Control ($L_2$ norm)', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(left=0); ax.set_ylim(bottom=0)

        plt.tight_layout()
        self._save(fig, 'rmi_divB.png')
        print(f"  ✓ divB comparison")

    def plot_all(self):
        print("\n=== Generating figures ===")
        sys.stdout.flush()
        self.plot_density_comparison()
        self.plot_passive_scalar()
        self.plot_schlieren()
        self.plot_vorticity_comparison()
        self.plot_magnetic_pressure_with_fieldlines()
        self.plot_diagnostics()
        self.plot_evolution()
        self.plot_hero_enstrophy()
        self.plot_hero_density()
        self.plot_interface_shape()
        self.plot_spectral_modes()
        self.plot_summary_bars()
        self.plot_mixing_width_hero()
        self.plot_linear_theory_comparison()
        self.plot_interface_perturbation()
        self.plot_stagnation_pressure()
        self.plot_divB_comparison()

        print(f"\n=== Figure manifest ({len(self._saved_files)} files) ===")
        total_kb = 0
        for fname, ok, sz in self._saved_files:
            status = '✓' if ok else '✗'
            print(f"  {status} {fname:40s}  {sz:7.0f} KB")
            total_kb += sz
        print(f"  {'Total':>42s}  {total_kb:7.0f} KB")
        print(f"  Directory: {os.path.abspath(OUTPUT_DIR)}")
        print("=== All figures done ===\n")
        sys.stdout.flush()
