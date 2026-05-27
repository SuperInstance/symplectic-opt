//! Conservation law tracking during optimization.

/// A conservation law: a quantity that should remain constant during dynamics.
pub trait ConservationLaw {
    /// Name of the conserved quantity.
    fn name(&self) -> &str;
    /// Compute the quantity from state (q, p).
    fn compute(&self, q: &[f64], p: &[f64]) -> f64;
    /// Check if conservation is satisfied within tolerance.
    fn is_conserved(&self, initial: f64, current: f64, tolerance: f64) -> bool {
        (initial - current).abs() < tolerance
    }
}

/// Energy conservation tracker.
pub struct EnergyTracker {
    /// Initial energy value.
    pub initial_energy: f64,
    /// History of energy values.
    pub history: Vec<f64>,
    /// Tolerance for conservation check.
    pub tolerance: f64,
}

impl EnergyTracker {
    /// Create a new tracker with initial state.
    pub fn new(
        q: &[f64],
        p: &[f64],
        kinetic_fn: fn(&[f64]) -> f64,
        potential_fn: fn(&[f64]) -> f64,
    ) -> Self {
        let energy = kinetic_fn(p) + potential_fn(q);
        Self {
            initial_energy: energy,
            history: vec![energy],
            tolerance: 0.01,
        }
    }

    /// Create a tracker from a known initial energy.
    pub fn from_energy(energy: f64) -> Self {
        Self {
            initial_energy: energy,
            history: vec![energy],
            tolerance: 0.01,
        }
    }

    /// Record energy at a new time step.
    pub fn record(&mut self, energy: f64) {
        self.history.push(energy);
    }

    /// Maximum energy drift from initial.
    pub fn max_drift(&self) -> f64 {
        self.history
            .iter()
            .map(|e| (e - self.initial_energy).abs())
            .fold(0.0f64, f64::max)
    }

    /// Average energy drift.
    pub fn avg_drift(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        self.history
            .iter()
            .map(|e| (e - self.initial_energy).abs())
            .sum::<f64>()
            / self.history.len() as f64
    }

    /// Whether energy is conserved throughout the history.
    pub fn is_conserved(&self) -> bool {
        self.history
            .iter()
            .all(|e| (e - self.initial_energy).abs() < self.tolerance)
    }
}

/// Momentum conservation: total momentum should be constant.
pub struct MomentumConservation {
    pub initial_momentum: Vec<f64>,
}

impl MomentumConservation {
    pub fn new(p: &[f64]) -> Self {
        Self {
            initial_momentum: p.to_vec(),
        }
    }

    /// Check if momentum is conserved.
    pub fn is_conserved(&self, p: &[f64], tolerance: f64) -> bool {
        if p.len() != self.initial_momentum.len() {
            return false;
        }
        p.iter()
            .zip(&self.initial_momentum)
            .all(|(a, b)| (a - b).abs() < tolerance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_tracker_initial() {
        let tracker = EnergyTracker::from_energy(5.0);
        assert_eq!(tracker.initial_energy, 5.0);
        assert_eq!(tracker.history.len(), 1);
    }

    #[test]
    fn test_energy_tracker_record() {
        let mut tracker = EnergyTracker::from_energy(5.0);
        tracker.record(5.01);
        tracker.record(4.99);
        assert_eq!(tracker.history.len(), 3);
    }

    #[test]
    fn test_energy_tracker_conserved() {
        let mut tracker = EnergyTracker::from_energy(5.0);
        tracker.record(5.001);
        tracker.record(4.999);
        tracker.record(5.002);
        assert!(tracker.is_conserved());
    }

    #[test]
    fn test_energy_tracker_not_conserved() {
        let mut tracker = EnergyTracker::from_energy(5.0);
        tracker.record(5.0);
        tracker.record(6.0); // Big jump
        assert!(!tracker.is_conserved());
    }

    #[test]
    fn test_max_drift() {
        let mut tracker = EnergyTracker::from_energy(5.0);
        tracker.record(5.1);
        tracker.record(4.8);
        assert!((tracker.max_drift() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_momentum_conservation() {
        let mc = MomentumConservation::new(&[1.0, 2.0, 3.0]);
        assert!(mc.is_conserved(&[1.0, 2.0, 3.0], 0.01));
        assert!(!mc.is_conserved(&[1.0, 2.0, 4.0], 0.01));
    }
}
