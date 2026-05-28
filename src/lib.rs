#![allow(
    clippy::needless_range_loop,
    clippy::new_without_default,
    clippy::type_complexity,
    dead_code
)]
//! # Symplectic Opt
//!
//! Symplectic optimization and Hamiltonian dynamics for conservation-law-aware training.
//!
//! # Key Concepts
//! - **Symplectic matrices**: 2n×2n matrices preserving the canonical symplectic form
//! - **Hamiltonian systems**: H(q,p) = T(p) + V(q), separable into kinetic and potential energy
//! - **Symplectic integrators**: Numerical methods that exactly preserve symplectic structure
//! - **Conservation laws**: Energy, momentum, angular momentum tracking during optimization

mod conservation;
mod hamiltonian;
mod natural_gradient;
mod symplectic;

pub use conservation::{ConservationLaw, EnergyTracker};
pub use hamiltonian::{HamiltonianSystem, SeparableHamiltonian};
pub use natural_gradient::NaturalGradient;
pub use symplectic::SymplecticMatrix;
