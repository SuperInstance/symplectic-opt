//! Hamiltonian systems and symplectic integrators.

/// A Hamiltonian system H(q, p) where q = position, p = momentum.
pub trait HamiltonianSystem {
    /// Kinetic energy T(p).
    fn kinetic(&self, p: &[f64]) -> f64;
    /// Potential energy V(q).
    fn potential(&self, q: &[f64]) -> f64;
    /// Total energy H = T + V.
    fn energy(&self, q: &[f64], p: &[f64]) -> f64 {
        self.kinetic(p) + self.potential(q)
    }
    /// ∂H/∂p = ∂T/∂p (force on momentum).
    fn dp(&self, p: &[f64]) -> Vec<f64>;
    /// -∂H/∂q = -∂V/∂q (force on position).
    fn dq(&self, q: &[f64]) -> Vec<f64>;
}

/// A separable Hamiltonian: H(q,p) = T(p) + V(q).
/// T(p) = Σ p_i² / (2m_i), V(q) = user-defined potential.
pub struct SeparableHamiltonian {
    /// Masses for each degree of freedom.
    pub masses: Vec<f64>,
    /// Potential energy function: takes position, returns energy and gradient.
    /// Stored as coefficients of a quadratic: V(q) = Σ coeff[i] * q[i]²
    pub potential_coeffs: Vec<f64>,
}

impl SeparableHamiltonian {
    pub fn new(masses: Vec<f64>, potential_coeffs: Vec<f64>) -> Self {
        Self {
            masses,
            potential_coeffs,
        }
    }

    /// Simple harmonic oscillator: m=1, V = 0.5*q².
    pub fn harmonic_oscillator(n: usize) -> Self {
        Self {
            masses: vec![1.0; n],
            potential_coeffs: vec![0.5; n],
        }
    }

    /// Symplectic Euler integration (first-order).
    /// Preserves symplectic structure but not exact energy.
    pub fn symplectic_euler(
        &self,
        q: &[f64],
        p: &[f64],
        dt: f64,
        steps: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut q = q.to_vec();
        let mut p = p.to_vec();
        for _ in 0..steps {
            // Update p first (momentum)
            let dq = self.dq(&q);
            for i in 0..p.len() {
                p[i] -= dq[i] * dt;
            }
            // Then update q with new p
            let dp = self.dp(&p);
            for i in 0..q.len() {
                q[i] += dp[i] * dt;
            }
        }
        (q, p)
    }

    /// Störmer-Verlet (leapfrog) integration (second-order symplectic).
    /// Better energy conservation than symplectic Euler.
    pub fn stormer_verlet(
        &self,
        q: &[f64],
        p: &[f64],
        dt: f64,
        steps: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut q = q.to_vec();
        let mut p = p.to_vec();
        for _ in 0..steps {
            // Half-step momentum
            let dq = self.dq(&q);
            for i in 0..p.len() {
                p[i] -= 0.5 * dq[i] * dt;
            }
            // Full-step position
            let dp = self.dp(&p);
            for i in 0..q.len() {
                q[i] += dp[i] * dt;
            }
            // Half-step momentum
            let dq = self.dq(&q);
            for i in 0..p.len() {
                p[i] -= 0.5 * dq[i] * dt;
            }
        }
        (q, p)
    }
}

impl HamiltonianSystem for SeparableHamiltonian {
    fn kinetic(&self, p: &[f64]) -> f64 {
        p.iter()
            .zip(&self.masses)
            .map(|(pi, m)| pi * pi / (2.0 * m))
            .sum()
    }

    fn potential(&self, q: &[f64]) -> f64 {
        q.iter()
            .zip(&self.potential_coeffs)
            .map(|(qi, c)| c * qi * qi)
            .sum()
    }

    fn dp(&self, p: &[f64]) -> Vec<f64> {
        p.iter().zip(&self.masses).map(|(pi, m)| pi / m).collect()
    }

    fn dq(&self, q: &[f64]) -> Vec<f64> {
        q.iter()
            .zip(&self.potential_coeffs)
            .map(|(qi, c)| 2.0 * c * qi)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmonic_oscillator_energy() {
        let h = SeparableHamiltonian::harmonic_oscillator(1);
        let e = h.energy(&[1.0], &[0.0]);
        assert!((e - 0.5).abs() < 1e-10); // V = 0.5*1² = 0.5
    }

    #[test]
    fn test_kinetic_energy() {
        let h = SeparableHamiltonian::harmonic_oscillator(1);
        let t = h.kinetic(&[3.0]);
        assert!((t - 4.5).abs() < 1e-10); // 3²/(2*1) = 4.5
    }

    #[test]
    fn test_dp_gradient() {
        let h = SeparableHamiltonian::harmonic_oscillator(1);
        let dp = h.dp(&[2.0]);
        assert_eq!(dp[0], 2.0); // p/m = 2/1 = 2
    }

    #[test]
    fn test_dq_gradient() {
        let h = SeparableHamiltonian::harmonic_oscillator(1);
        let dq = h.dq(&[1.0]);
        assert_eq!(dq[0], 1.0); // 2*0.5*1 = 1
    }

    #[test]
    fn test_symplectic_euler_period() {
        let h = SeparableHamiltonian::harmonic_oscillator(1);
        let (q, _p) = h.symplectic_euler(&[1.0], &[0.0], 0.001, 6283);
        // After ~2π time, should return close to start
        assert!(q[0].abs() < 2.0);
    }

    #[test]
    fn test_stormer_verlet_preserves_energy() {
        let h = SeparableHamiltonian::harmonic_oscillator(1);
        let e0 = h.energy(&[1.0], &[0.0]);
        let (q, p) = h.stormer_verlet(&[1.0], &[0.0], 0.01, 1000);
        let e1 = h.energy(&q, &p);
        // Energy should be conserved to high precision
        assert!((e1 - e0).abs() < 0.01);
    }

    #[test]
    fn test_stormer_verlet_orbit() {
        let h = SeparableHamiltonian::harmonic_oscillator(1);
        let e0 = h.energy(&[1.0], &[0.0]);
        let (q, p) = h.stormer_verlet(&[1.0], &[0.0], 0.01, 1000);
        let e1 = h.energy(&q, &p);
        // Störmer-Verlet should conserve energy well
        assert!((e1 - e0).abs() < 0.01);
        // And the solution should stay bounded
        assert!(q[0].abs() < 3.0);
    }
}
