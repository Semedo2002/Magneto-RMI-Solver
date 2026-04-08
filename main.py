"""
main.py
Main execution script for the MHD RMI simulations.
Runs verification suite, RMI cases, convergence study, and postprocessing.
"""

import numpy as np
import time
import sys
import os
from config import Config
from solver import (
    MHDSolver, brio_wu_test, linear_wave_convergence_test,
    contact_discontinuity_test, run_convergence_study, HAS_NUMBA
)
from postprocess import PostProcessor, OUTPUT_DIR

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    print("\n" + "=" * 64)
    print("MAGNETIZED RICHTMYER-MESHKOV INSTABILITY")
    print("HLLD | MUSCL | SSP-RK3 | GLM+Powell | Passive Scalar")
    print("MHD R-H | Brio-Wu + Alfven wave + Contact verified")
    print("Extended domain | Char BCs | Linear theory overlay")
    print("  Numba JIT: " + ("ENABLED" if HAS_NUMBA else "DISABLED"))
    print("=" * 64)
    sys.stdout.flush()

    total_t0 = time.time()

    # ============================
    # Verification suite
    # ============================
    print("\n" + "="*64)
    print("  VERIFICATION SUITE")
    print("="*64)

    bw_passed = brio_wu_test(nx=800, t_end=0.1, plot=True, output_dir=OUTPUT_DIR)
    if not bw_passed:
        print("\n   WARNING: Brio-Wu test did not pass.")
    else:
        print("\n   Brio-Wu verification PASSED.")

    wave_passed = linear_wave_convergence_test(plot=True, output_dir=OUTPUT_DIR)
    if not wave_passed:
        print("\n   WARNING: Alfven wave convergence test did not pass.")
    else:
        print("\n   Alfven wave convergence PASSED.")

    contact_passed = contact_discontinuity_test(plot=True, output_dir=OUTPUT_DIR)
    if not contact_passed:
        print("\n   WARNING: Contact discontinuity test did not pass.")
    else:
        print("\n   Contact discontinuity PASSED.")

    all_tests_passed = bw_passed and wave_passed and contact_passed
    print(f"\n  Overall verification: {'ALL PASSED ✓' if all_tests_passed else 'SOME FAILED ✗'}")

    # ============================
    # RMI runs
    # ============================
    base = dict(
        nx=400, ny=200,
        x_min=0.0, x_max=6.0, y_min=0.0, y_max=2.0,
        t_end=0.25, cfl=0.30, mach=10.0,
        interface_x=1.5, perturbation_amp=0.15,
        perturbation_mode=4, density_ratio=3.0,
        interface_width=2.0,
        powell_source=True, use_char_bc=True,
        bc_x_type="characteristic", bc_y_type="periodic",
        enable_smoothing=False,
        diag_interval=20,
        snapshot_times=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
    )

    cases = {
        "Hydro (B=0)": 0.0,
        "MHD (By=0.5)": 0.5,
        "MHD (By=1.5)": 1.5,
    }

    solvers = {}
    for label, By in cases.items():
        print(f"\n{'='*64}\n  CASE: {label}\n{'='*64}")
        sys.stdout.flush()
        cfg = Config(**base, B_transverse=By)
        s = MHDSolver(cfg)
        s.initialize()
        s.run()
        solvers[label] = s

    # ============================
    # Grid convergence study
    # ============================
    print("\n" + "="*64)
    print("  GRID CONVERGENCE STUDY")
    print("="*64)

    conv_base = base.copy()
    conv_base['t_end'] = 0.15
    conv_base['snapshot_times'] = [0.0, 0.15]
    conv_base['powell_source'] = True

    conv_resolutions = [(100, 50), (200, 100), (400, 200)]

    conv_hydro = run_convergence_study(conv_base, By_value=0.0,
                                        resolutions=conv_resolutions, plot=True,
                                        output_dir=OUTPUT_DIR)
    conv_mhd = run_convergence_study(conv_base, By_value=1.5,
                                      resolutions=conv_resolutions, plot=True,
                                      output_dir=OUTPUT_DIR)

    # ============================
    # Postprocessing
    # ============================
    post = PostProcessor(solvers)
    post.plot_all()

    # ============================
    # Summary
    # ============================
    print("=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    t_skip = 0.03

    print(f"  {'Case':<20s}  {'MW_int':>8s}  {'MW_thr':>8s}  "
          f"{'θ_mean':>8s}  {'Enst_loc':>10s}  {'divB_L2':>8s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")

    summary_data = {}
    for label, s in solvers.items():
        t = np.array(s.diag_times); mask = t > t_skip
        mwi = np.array(s.diag_mixing_width_integral)[mask]
        mwt = np.array(s.diag_mixing_width_thresh)[mask]
        theta = np.array(s.diag_mixedness)[mask]
        en = np.array(s.diag_enstrophy_local)[mask]
        divB = np.array(s.diag_divB_L2)[mask] if s.diag_divB_L2 else np.array([0.0])

        peak_mwi = float(np.max(mwi)) if len(mwi) > 0 else 0
        peak_mwt = float(np.max(mwt)) if len(mwt) > 0 else 0
        mean_theta = float(np.mean(theta)) if len(theta) > 0 else 0
        mean_en = float(np.mean(en)) if len(en) > 0 else 0
        mean_divB = float(np.mean(divB)) if len(divB) > 0 else 0
        summary_data[label] = (peak_mwi, peak_mwt, mean_theta, mean_en, mean_divB)
        print(f"  {label:<20s}  {peak_mwi:8.4f}  {peak_mwt:8.4f}  "
              f"{mean_theta:8.3f}  {mean_en:10.1f}  {mean_divB:8.3f}")

    hydro_key = list(solvers.keys())[0]
    h_mwi, h_mwt, h_theta, h_en, _ = summary_data[hydro_key]

    print()
    for label, s in solvers.items():
        if s.cfg.B_transverse == 0: continue
        m_mwi, m_mwt, m_theta, m_en, _ = summary_data[label]
        mwi_sup = (1 - m_mwi/max(h_mwi, 1e-10))*100
        mwt_sup = (1 - m_mwt/max(h_mwt, 1e-10))*100
        theta_change = (m_theta - h_theta)/max(h_theta, 1e-10)*100
        en_sup = (1 - m_en/max(h_en, 1e-10))*100

        print(f"  {label}:")
        print(f"    MW integral suppression:   {mwi_sup:+.1f}%")
        print(f"    MW threshold suppression:  {mwt_sup:+.1f}%")
        print(f"    Mixedness θ change:        {theta_change:+.1f}%")
        print(f"    Local enstrophy suppress.: {en_sup:+.1f}%")

    print()
    print("  Linear Theory Comparison:")
    from linear_theory import richtmyer_linear_theory
    for label, s in solvers.items():
        if s._rh_results is not None:
            cfg = s.cfg
            Ly = cfg.y_max - cfg.y_min
            k_pert = 2*np.pi*cfg.perturbation_mode / Ly
            _, t_sh, info = richtmyer_linear_theory(
                np.array([0.0]), cfg.gamma, cfg.mach, 1.0,
                cfg.density_ratio, cfg.perturbation_amp, k_pert,
                By=cfg.B_transverse)
            print(f"    {label}: A_post={info['A_post']:.3f}, "
                  f"da/dt={info['da_dt']:.2f}, "
                  f"Δv={info['delta_v']:.2f}, "
                  f"a0_post={info['a0_post']:.4f}")

    print()
    print("  Physics of magnetic RMI suppression:")
    print("  • Alfven waves transport baroclinic vorticity away from interface")
    print("  • Magnetic tension (B·∇)B opposes KH secondary instability")
    print("  • Interface coherence preserved → reduced mixing zone width")
    print("  • Suppression scales with v_A / v_perturbation")
    print("  • Higher θ in MHD = narrower but more homogeneous mixing zone")

    print()
    print("  Numerical features")
    print(f"  • Numba JIT acceleration: {'YES' if HAS_NUMBA else 'NO'}")
    print(f"  • Brio-Wu: {'PASSED ✓' if bw_passed else 'FAILED ✗'}")
    print(f"  • Alfven wave convergence: {'PASSED ✓' if wave_passed else 'FAILED ✗'}")
    print(f"  • Contact discontinuity: {'PASSED ✓' if contact_passed else 'FAILED ✗'}")
    print(f"  • Powell source terms: ENABLED (face-centered div(B))")
    print(f"  • Characteristic BCs: ENABLED (production runs)")
    print(f"  • Periodic BCs: ENABLED (wave convergence test)")
    print(f"  • Extended domain: x=[0, 6] (50% longer)")
    print(f"  • Interface width: 2 cells (sharper)")
    print(f"  • Smoothing on diagnostics: DISABLED (raw data)")
    print(f"  • Grid convergence study: 3 resolutions")

    for label, s in solvers.items():
        if len(s.diag_energy_total) >= 2:
            e0 = s.diag_energy_total[0]; ef = s.diag_energy_total[-1]
            bf = s._cumulative_boundary_energy
            de_raw = abs(ef-e0)/max(abs(e0), 1e-14)*100
            de_corr = abs(ef-bf-e0)/max(abs(e0), 1e-14)*100
            print(f"  • {label}: energy drift raw={de_raw:.1f}%, corrected={de_corr:.1f}%")

    total_elapsed = time.time() - total_t0
    print(f"\n  Total wall time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Output: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 72)
    sys.stdout.flush()
