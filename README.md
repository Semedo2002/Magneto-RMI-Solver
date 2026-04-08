Magnetized Richtmyer Meshkov Instability Solver
<img width="2945" height="870" alt="rmi_schlieren" src="https://github.com/user-attachments/assets/6de8058a-70c9-468c-a495-4a54f2b6c362" />

A 2D ideal Magnetohydrodynamics solver. This codebase simulates and analyzes the Richtmyer Meshkov Instability in magnetized plasmas, showing the suppression of baroclinic vorticity deposition via transverse magnetic fields.
<img width="2857" height="1490" alt="rmi_evolution" src="https://github.com/user-attachments/assets/d82db42c-227c-44d3-9d68-772b8f0d71e4" />

Numericals:
<img width="3216" height="890" alt="convergence_Hydro" src="https://github.com/user-attachments/assets/157ef67c-0006-4538-92f9-d4c18299206e" />

I tried to build this solver to handle high Mach shock interface interactions without crashing or diffusing the physics.

Riemann Solver: Harten Lax van Leer Discontinuities by Miyoshi & Kusano, ensuring exact resolution of contact discontinuities and Alfven waves.
Spatial Discretization: 2nd-Order MUSCL reconstruction with primitive variable minmod limiting to prevent spurious oscillations near shocks.
Time Integration: 3rd-Order Strong Stability Preserving SSP-RK3 for rigorous TVD compliance.
Divergence Cleaning: Mixed GLM advection and Powell's 8-wave source terms.
Fully vectorized and JIT compiled using Numba.

Verification:

The solver has a built in verification suite to prove the fidelity of the numerics before running the full RMI case:
1.Brio-Wu Shock Tube which validates the HLLD solver's handling of compound waves and shocks.
2.Alfvén Wave Convergence that confirms theoretical 2nd-order spatial accuracy.
3.Contact Discontinuity Test to ensures stationary contacts do not artificially diffuse over time.

Magnetic Suppression of the RMI:

The Richtmyer Meshkov Instability occurs when a shock wave impulsively accelerates a density interface. In pure hydrodynamics, this generates intense vorticity. 

In this simulation, I introduced a transverse magnetic field, because the magnetic field lines are frozen in to the plasma, the growing interface bends the field lines, generating a restoring magnetic tension force . The solver accurately captures how this tension strips vorticity away from the interface as traveling Alfvén waves, suppressing the  Kelvin Helmholtz roll ups.

The results of this solver show excellent agreement with Richtmyer's impulsive linear growth theory.

Installation & Usage:
The code relies on standard scientific Python libraries. numba is highly recommended for performance but will gracefully fall back to standard Python if not installed.
```bash
pip install numpy matplotlib numba
