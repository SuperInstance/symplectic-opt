//! # Symplectic Opt Tutorial
//!
//! A progressive introduction to symplectic matrices, Hamiltonian systems, and conservation-aware optimization.
//!
//! Run with: `cargo run --example tutorial`

use symplectic_opt::{
    EnergyTracker, HamiltonianSystem, NaturalGradient, SeparableHamiltonian,
    SymplecticMatrix,
};

fn main() {
    lesson_1_symplectic_identity_and_j();
    lesson_2_symplectic_matrices();
    lesson_3_hamiltonian_energy();
    lesson_4_symplectic_euler();
    lesson_5_stormer_verlet();
    lesson_6_energy_conservation_tracking();
    lesson_7_momentum_conservation();
    lesson_8_natural_gradient();
}

// ── Lesson 1: The Identity and the Symplectic Form J ────────────────────
fn lesson_1_symplectic_identity_and_j() {
    println!("═══════════════════════════════════════════");
    println!("Lesson 1: Symplectic Identity and the J Matrix");
    println!("═══════════════════════════════════════════");
    println!();

    // The 2n×2n identity is always symplectic.
    let id = SymplecticMatrix::identity(2);
    println!("4×4 identity is symplectic? {}", id.is_symplectic());
    println!("Half-dimension n = {}, full dimension = {}", id.n, id.dim());
    println!();

    // The canonical symplectic form J.
    // J = [0, I; -I, 0]
    let j = SymplecticMatrix::j_matrix(2);
    println!("J matrix (n=2):");
    for row in &j.data {
        println!("  [{:.0}, {:.0}, {:.0}, {:.0}]", row[0], row[1], row[2], row[3]);
    }
    println!("J is symplectic? {}", j.is_symplectic());
    println!();
}

// ── Lesson 2: Building and Composing Symplectic Matrices ────────────────
fn lesson_2_symplectic_matrices() {
    println!("═══════════════════════════════════════════");
    println!("Lesson 2: Building Symplectic Matrices");
    println!("═══════════════════════════════════════════");
    println!();

    // A 2×2 symplectic matrix (shear): M = [[1, s], [0, 1]].
    let shear = SymplecticMatrix::new(vec![vec![1.0, 0.5], vec![0.0, 1.0]]);
    match &shear {
        Ok(m) => {
            println!("2×2 shear [[1, 0.5], [0, 1]] is symplectic: {}", m.is_symplectic());
            let v = m.mul_vec(&[1.0, 1.0]);
            println!("  M × [1, 1] = [{:.1}, {:.1}]", v[0], v[1]);
        }
        Err(e) => println!("Error: {}", e),
    }
    println!();

    // A non-symplectic matrix is rejected.
    let bad = SymplecticMatrix::new(vec![vec![2.0, 0.0], vec![0.0, 2.0]]);
    println!("[[2,0],[0,2]] is symplectic? {:?}", bad.map(|_| "yes").unwrap_err());
    println!();

    // Matrix inverse: M⁻¹ = −J M^T J.
    let m = shear.unwrap();
    let inv = m.inverse();
    println!("Inverse of shear:");
    println!("  [{:.1}, {:.1}]", inv.data[0][0], inv.data[0][1]);
    println!("  [{:.1}, {:.1}]", inv.data[1][0], inv.data[1][1]);
    println!("  M⁻¹ is symplectic? {}", inv.is_symplectic());
    println!();
}

// ── Lesson 3: Hamiltonian Systems ───────────────────────────────────────
fn lesson_3_hamiltonian_energy() {
    println!("═══════════════════════════════════════════");
    println!("Lesson 3: Hamiltonian Energy");
    println!("═══════════════════════════════════════════");
    println!();

    // A simple harmonic oscillator: H = T(p) + V(q), with T = p²/2, V = ½q².
    let h = SeparableHamiltonian::harmonic_oscillator(1);

    let q = [1.0];
    let p = [0.0];
    println!("State: q = [1.0], p = [0.0]");
    println!("  Kinetic energy T(p) = {:.2}", h.kinetic(&p));
    println!("  Potential energy V(q) = {:.2}", h.potential(&q));
    println!("  Total energy H = {:.2}", h.energy(&q, &p));
    println!();

    // At maximum displacement, all energy is potential.
    // At equilibrium, all energy is kinetic.
    let q2 = [0.0];
    let p2 = [1.0];
    println!("State: q = [0.0], p = [1.0]");
    println!("  T = {:.2}, V = {:.2}, H = {:.2}", h.kinetic(&p2), h.potential(&q2), h.energy(&q2, &p2));
    println!();

    // Gradients.
    println!("∂H/∂p at p=[2.0]: {:?}  (velocity = p/m)", h.dp(&[2.0]));
    println!("∂V/∂q at q=[1.0]: {:?}  (restoring force)", h.dq(&[1.0]));
    println!();
}

// ── Lesson 4: Symplectic Euler Integration ──────────────────────────────
fn lesson_4_symplectic_euler() {
    println!("═══════════════════════════════════════════");
    println!("Lesson 4: Symplectic Euler (1st Order)");
    println!("═══════════════════════════════════════════");
    println!();

    let h = SeparableHamiltonian::harmonic_oscillator(1);
    let q0 = [1.0];
    let p0 = [0.0];
    let e0 = h.energy(&q0, &p0);
    println!("Initial: q = [1.0], p = [0.0], H = {:.4}", e0);
    println!();

    // Integrate for one quarter period (≈ π/2 time units).
    let dt = 0.01;
    let steps = 157; // ≈ 1.57 seconds ≈ π/2
    let (q, p) = h.symplectic_euler(&q0, &p0, dt, steps);
    let e1 = h.energy(&q, &p);
    println!("After ~π/2 time (symplectic Euler, dt={}):", dt);
    println!("  q = [{:.4}], p = [{:.4}]", q[0], p[0]);
    println!("  H = {:.4} (drift = {:.6})", e1, (e1 - e0).abs());
    println!();

    // Larger time step shows more drift but orbit stays bounded.
    let (q2, p2) = h.symplectic_euler(&q0, &p0, 0.1, 16);
    let e2 = h.energy(&q2, &p2);
    println!("Larger dt=0.1: H = {:.4} (drift = {:.4})", e2, (e2 - e0).abs());
    println!();
}

// ── Lesson 5: Störmer-Verlet Integration ────────────────────────────────
fn lesson_5_stormer_verlet() {
    println!("═══════════════════════════════════════════");
    println!("Lesson 5: Störmer-Verlet (2nd Order)");
    println!("═══════════════════════════════════════════");
    println!();

    let h = SeparableHamiltonian::harmonic_oscillator(1);
    let q0 = [1.0];
    let p0 = [0.0];
    let e0 = h.energy(&q0, &p0);

    // Störmer-Verlet is 2nd order — much better energy conservation.
    let dt = 0.01;
    let (q, p) = h.stormer_verlet(&q0, &p0, dt, 1000);
    let e1 = h.energy(&q, &p);
    println!("Störmer-Verlet, dt=0.01, 1000 steps:");
    println!("  q = [{:.6}], p = [{:.6}]", q[0], p[0]);
    println!("  H = {:.6} (drift = {:.2e})", e1, (e1 - e0).abs());
    println!();

    // Compare with symplectic Euler at same step count.
    let (qe, pe) = h.symplectic_euler(&q0, &p0, dt, 1000);
    let ee = h.energy(&qe, &pe);
    println!("Symplectic Euler for comparison:");
    println!("  H = {:.6} (drift = {:.2e})", ee, (ee - e0).abs());
    println!("  → Verlet has much smaller energy drift!");
    println!();
}

// ── Lesson 6: Energy Conservation Tracking ──────────────────────────────
fn lesson_6_energy_conservation_tracking() {
    println!("═══════════════════════════════════════════");
    println!("Lesson 6: Energy Conservation Tracking");
    println!("═══════════════════════════════════════════");
    println!();

    let h = SeparableHamiltonian::harmonic_oscillator(1);
    let q0 = [1.0];
    let p0 = [0.0];
    let e0 = h.energy(&q0, &p0);

    // Track energy over multiple Störmer-Verlet steps.
    let mut tracker = EnergyTracker::from_energy(e0);
    let dt = 0.01;
    let mut q = q0.to_vec();
    let mut p = p0.to_vec();
    for step in 1..=20 {
        let (qn, pn) = h.stormer_verlet(&q, &p, dt, 50);
        q = qn;
        p = pn;
        tracker.record(h.energy(&q, &p));
        if step % 5 == 0 {
            println!("  Step {:>2} (t={:.1}): H = {:.8}", step, step as f64 * 50.0 * dt, h.energy(&q, &p));
        }
    }
    println!();
    println!("Max energy drift: {:.2e}", tracker.max_drift());
    println!("Avg energy drift: {:.2e}", tracker.avg_drift());
    println!("Energy conserved (tol=0.01)? {}", tracker.is_conserved());
    println!();
}

// ── Lesson 7: Momentum Conservation ─────────────────────────────────────
fn lesson_7_momentum_conservation() {
    println!("═══════════════════════════════════════════");
    println!("Lesson 7: Momentum Conservation");
    println!("═══════════════════════════════════════════");
    println!();

    // For a multi-body system with zero total momentum, it should stay zero.
    let h = SeparableHamiltonian::new(vec![1.0, 1.0], vec![0.5, 0.5]);
    let q0 = [1.0, -1.0];
    let p0 = [1.0, -1.0]; // total momentum = 0

    let initial_total_p: f64 = p0.iter().sum();
    println!("Initial momentum: {:?}", p0);
    println!("Total momentum: {:.1}", initial_total_p);
    println!();

    let (q, p) = h.stormer_verlet(&q0, &p0, 0.01, 100);
    let final_total_p: f64 = p.iter().sum();
    println!("After 100 Störmer-Verlet steps:");
    println!("  q = [{:.4}, {:.4}]", q[0], q[1]);
    println!("  p = [{:.4}, {:.4}]", p[0], p[1]);
    println!("  Total momentum: {:.4}", final_total_p);
    println!("  Momentum conserved? {}", (final_total_p - initial_total_p).abs() < 0.01);
    println!();

    // Demonstrate the ConservationLaw trait via energy.
    println!("ConservationLaw trait check:");
    let e0 = h.energy(&q0, &p0);
    let e1 = h.energy(&q, &p);
    println!("  Energy drift: {:.2e}", (e1 - e0).abs());
    println!("  Is conserved (tol=0.01)? {}", (e1 - e0).abs() < 0.01);
    println!();
}

// ── Lesson 8: Natural Gradient Descent ──────────────────────────────────
fn lesson_8_natural_gradient() {
    println!("═══════════════════════════════════════════");
    println!("Lesson 8: Natural Gradient Descent");
    println!("═══════════════════════════════════════════");
    println!();

    // Natural gradient uses the Fisher information matrix as a metric.
    let ng = NaturalGradient::new(0.1);
    println!("Natural gradient: lr = {}", ng.lr);
    println!();

    // Build a Fisher information matrix from gradient samples.
    let gradients = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let fisher = NaturalGradient::fisher_matrix(&gradients);
    println!("Fisher matrix from 3 gradient samples:");
    for row in &fisher {
        println!("  [{:.4}, {:.4}]", row[0], row[1]);
    }
    println!();

    // One natural gradient step.
    let params = vec![5.0, 5.0];
    let gradient = vec![1.0, 1.0];
    let new_params = ng.step(&params, &fisher, &gradient);
    println!("Before step: {:?}", params);
    println!("After step:  [{:.4}, {:.4}]", new_params[0], new_params[1]);
    println!();

    // KL divergence between two distributions.
    let mu1 = vec![1.0, 0.0];
    let mu2 = vec![0.0, 0.0];
    let sigma_inv = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let kl = NaturalGradient::kl_divergence(&mu1, &mu2, &sigma_inv);
    println!("KL(N([1,0], I) || N([0,0], I)) = {:.4}", kl);
    println!();
}
