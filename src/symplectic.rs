//! Symplectic matrices: 2n×2n matrices satisfying M^T J M = J.

/// A 2n×2n symplectic matrix.
///
/// Satisfies the condition: M^T J M = J where J is the standard symplectic form:
/// J = [0, I; -I, 0]
#[derive(Debug, Clone)]
pub struct SymplecticMatrix {
    /// The matrix entries, stored as flat 2n×2n.
    pub data: Vec<Vec<f64>>,
    /// Half-dimension n (matrix is 2n×2n).
    pub n: usize,
}

impl SymplecticMatrix {
    /// Create from a 2n×2n matrix, verifying symplecticity.
    pub fn new(data: Vec<Vec<f64>>) -> Result<Self, String> {
        let m = data.len();
        if m == 0 || !m.is_multiple_of(2) {
            return Err("Matrix must be 2n×2n".into());
        }
        for row in &data {
            if row.len() != m {
                return Err("Matrix must be square".into());
            }
        }
        let n = m / 2;
        let sm = Self { data, n };
        if !sm.is_symplectic() {
            return Err("Matrix is not symplectic: M^T J M ≠ J".into());
        }
        Ok(sm)
    }

    /// The 2n×2n identity (always symplectic).
    pub fn identity(n: usize) -> Self {
        let dim = 2 * n;
        Self {
            data: (0..dim)
                .map(|i| (0..dim).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
                .collect(),
            n,
        }
    }

    /// Standard symplectic form J.
    pub fn j_matrix(n: usize) -> Self {
        let dim = 2 * n;
        let mut j = vec![vec![0.0; dim]; dim];
        for i in 0..n {
            j[i][n + i] = 1.0;
            j[n + i][i] = -1.0;
        }
        Self { data: j, n }
    }

    /// Check symplecticity: M^T J M = J (within tolerance).
    pub fn is_symplectic(&self) -> bool {
        let j = Self::j_matrix(self.n);
        let mtjm = self.transpose().mul(&j).mul(self);
        let expected = Self::j_matrix(self.n);
        for i in 0..2 * self.n {
            for k in 0..2 * self.n {
                if (mtjm.data[i][k] - expected.data[i][k]).abs() > 1e-6 {
                    return false;
                }
            }
        }
        true
    }

    /// Matrix transpose.
    pub fn transpose(&self) -> Self {
        let dim = 2 * self.n;
        Self {
            data: (0..dim)
                .map(|i| (0..dim).map(|j| self.data[j][i]).collect())
                .collect(),
            n: self.n,
        }
    }

    /// Matrix multiplication.
    pub fn mul(&self, other: &Self) -> Self {
        let dim = 2 * self.n;
        let mut result = vec![vec![0.0; dim]; dim];
        for i in 0..dim {
            for k in 0..dim {
                for j in 0..dim {
                    result[i][k] += self.data[i][j] * other.data[j][k];
                }
            }
        }
        Self {
            data: result,
            n: self.n,
        }
    }

    /// Matrix-vector multiplication.
    pub fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        let dim = 2 * self.n;
        (0..dim)
            .map(|i| (0..dim).map(|j| self.data[i][j] * v[j]).sum())
            .collect()
    }

    /// Inverse of a symplectic matrix: M^{-1} = -J M^T J.
    pub fn inverse(&self) -> Self {
        let j = Self::j_matrix(self.n);
        let mt = self.transpose();
        let neg_j_mt_j = j.transpose().mul(&mt).mul(&j);
        let dim = 2 * self.n;
        Self {
            data: (0..dim)
                .map(|i| (0..dim).map(|k| -neg_j_mt_j.data[i][k]).collect())
                .collect(),
            n: self.n,
        }
    }

    /// Dimension of the matrix (2n).
    pub fn dim(&self) -> usize {
        2 * self.n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_is_symplectic() {
        let m = SymplecticMatrix::identity(2);
        assert!(m.is_symplectic());
    }

    #[test]
    fn test_j_matrix_is_symplectic() {
        let j = SymplecticMatrix::j_matrix(2);
        assert!(j.is_symplectic());
    }

    #[test]
    fn test_inverse_is_symplectic() {
        let m = SymplecticMatrix::identity(1);
        assert!(m.is_symplectic());
        let inv = m.inverse();
        assert!(inv.is_symplectic());
    }

    #[test]
    fn test_mul_vec() {
        let m = SymplecticMatrix::identity(1);
        let v = vec![3.0, 4.0];
        let mv = m.mul_vec(&v);
        assert_eq!(mv[0], 3.0);
        assert_eq!(mv[1], 4.0);
    }

    #[test]
    fn test_non_symplectic_rejected() {
        let data = vec![vec![2.0, 0.0], vec![0.0, 2.0]];
        assert!(SymplecticMatrix::new(data).is_err());
    }

    #[test]
    fn test_symplectic_2x2() {
        let data = vec![vec![1.0, 0.5], vec![0.0, 1.0]];
        let m = SymplecticMatrix::new(data);
        assert!(m.is_ok());
        assert!(m.unwrap().is_symplectic());
    }
}
