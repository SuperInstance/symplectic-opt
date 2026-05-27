//! Natural gradient descent using Fisher information matrix.

/// Natural gradient descent on Riemannian manifolds.
///
/// Uses the Fisher information matrix as the Riemannian metric,
/// providing the "steepest" descent direction in probability space.
pub struct NaturalGradient {
    /// Learning rate.
    pub lr: f64,
    /// Damping factor for Fisher matrix inversion (for numerical stability).
    pub damping: f64,
}

impl NaturalGradient {
    pub fn new(lr: f64) -> Self {
        Self { lr, damping: 1e-4 }
    }

    /// Compute the Fisher information matrix from gradients.
    /// F = E[∇log p(x|θ) ∇log p(x|θ)^T]
    /// Simplified: use outer product of gradient vectors.
    pub fn fisher_matrix(gradients: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if gradients.is_empty() {
            return vec![];
        }
        let n = gradients[0].len();
        let mut fisher = vec![vec![0.0; n]; n];
        for g in gradients {
            for i in 0..n {
                for j in 0..n {
                    fisher[i][j] += g[i] * g[j];
                }
            }
        }
        let m = gradients.len() as f64;
        for row in &mut fisher {
            for v in row.iter_mut() {
                *v /= m;
            }
        }
        fisher
    }

    /// Compute natural gradient: F^{-1} ∇L.
    /// Uses damped least squares: (F + λI)^{-1} g.
    pub fn natural_gradient(&self, fisher: &[Vec<f64>], gradient: &[f64]) -> Vec<f64> {
        let n = gradient.len();
        if n == 0 {
            return vec![];
        }

        // Add damping: (F + λI)
        let mut f_damped = fisher.to_vec();
        for i in 0..n {
            f_damped[i][i] += self.damping;
        }

        // Solve (F + λI) x = g using Gauss-Seidel iteration
        let mut x = gradient.to_vec();
        for _ in 0..20 {
            for i in 0..n {
                let mut sum = gradient[i];
                for j in 0..n {
                    if j != i {
                        sum -= f_damped[i][j] * x[j];
                    }
                }
                x[i] = sum / f_damped[i][i];
            }
        }

        x
    }

    /// Perform one natural gradient step.
    /// Returns updated parameters.
    pub fn step(&self, params: &[f64], fisher: &[Vec<f64>], gradient: &[f64]) -> Vec<f64> {
        let ng = self.natural_gradient(fisher, gradient);
        params
            .iter()
            .zip(ng.iter())
            .map(|(p, g)| p - self.lr * g)
            .collect()
    }

    /// KL divergence between two Gaussian approximations.
    /// KL(N(μ₁,Σ₁) || N(μ₂,Σ₂)) ≈ 0.5 * (μ₁-μ₂)^T Σ₂^{-1} (μ₁-μ₂)
    pub fn kl_divergence(mu1: &[f64], mu2: &[f64], sigma_inv: &[Vec<f64>]) -> f64 {
        let diff: Vec<f64> = mu1.iter().zip(mu2).map(|(a, b)| a - b).collect();
        let mut kl = 0.0;
        for i in 0..diff.len() {
            for j in 0..diff.len() {
                kl += diff[i] * sigma_inv[i][j] * diff[j];
            }
        }
        0.5 * kl
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fisher_matrix() {
        let gradients = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let fisher = NaturalGradient::fisher_matrix(&gradients);
        assert_eq!(fisher.len(), 2);
        // F = average of outer products
        // [[1,0],[0,0]] + [[0,0],[0,1]] + [[1,1],[1,1]] all / 3
        assert!((fisher[0][0] - 2.0 / 3.0).abs() < 1e-10);
        assert!((fisher[0][1] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_natural_gradient() {
        let ng = NaturalGradient::new(0.01);
        let fisher = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let gradient = vec![1.0, 2.0];
        let result = ng.natural_gradient(&fisher, &gradient);
        // With identity Fisher, natural gradient = gradient
        assert!((result[0] - 1.0).abs() < 0.1);
        assert!((result[1] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_step() {
        let ng = NaturalGradient::new(0.1);
        let fisher = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let gradient = vec![1.0, 1.0];
        let params = vec![5.0, 5.0];
        let new_params = ng.step(&params, &fisher, &gradient);
        assert!(new_params[0] < params[0]); // Should decrease
    }

    #[test]
    fn test_kl_divergence() {
        let mu1 = vec![1.0, 0.0];
        let mu2 = vec![0.0, 0.0];
        let sigma_inv = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let kl = NaturalGradient::kl_divergence(&mu1, &mu2, &sigma_inv);
        assert!((kl - 0.5).abs() < 1e-10); // 0.5 * 1^2 = 0.5
    }
}
