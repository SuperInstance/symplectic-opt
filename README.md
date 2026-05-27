# symplectic-opt

Symplectic optimization and Hamiltonian dynamics for conservation-law-aware training.

## Usage

```rust
use symplectic_opt::{SymplecticMatrix, SeparableHamiltonian};

// Symplectic matrices preserve the canonical form J
let m = SymplecticMatrix::identity(2);
assert!(m.is_symplectic());

// Separable Hamiltonian: H(q,p) = T(p) + V(q)
let h = SeparableHamiltonian::harmonic_oscillator(1);

// Störmer-Verlet integrator preserves energy
let (q, p) = h.stormer_verlet(&[1.0], &[0.0], 0.01, 1000);
```

## Features

- **Symplectic matrices**: 2n×2n with M^T J M = J verification
- **Hamiltonian systems**: Separable H(q,p) = T(p) + V(q)
- **Symplectic Euler** (1st order) and **Störmer-Verlet** (2nd order) integrators
- **Conservation law tracking**: Energy and momentum monitoring
- **Natural gradient descent**: Fisher information matrix optimization on Riemannian manifolds

## Tests

23 tests, all passing. `cargo test` to run.

## License

MIT
