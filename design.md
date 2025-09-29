# mdarray-random Design Document

## Overview

`mdarray-random` is a Rust crate built on top of `mdarray` for generating multi-dimensional
arrays of random numbers. It aims to provide a trait-based system
for various probability distributions and efficient sampling
operations.

## Core Architecture

### Distribution Trait

Every probability distribution implements the `Distribution<T>` trait with the following methods:

```rust
pub trait Distribution<T> {
    /// Draw a single value using the provided RNG
    fn sample<R: rand::RngCore + ?Sized>(&self, rng: &mut R) -> T;
    
    /// Draw elements with specified shape and return a DTensor (or Vec<T>)
    fn sample_tensor<R: rand::RngCore + ?Sized>(&self, rng: &mut R, shape: &[usize]) -> DTensor;
    
    /// Adapt sampling to a DTensor template (copy dtype/shape)
    fn sample_from_template<R: rand::RngCore + ?Sized>(
        &self, 
        rng: &mut R, 
        template: &DTensor
    ) -> Result<DTensor, RandomError>;
}
```

### Generator Structure

```rust
pub struct Generator<R: RandCore = rand::rngs::SmallRng> {
    rng: R,
}

impl Generator {
    pub fn from_seed(seed: u64) -> Self { ... }
    
    pub fn seed(&mut self, seed: u64) { ... }
    
    pub fn randchoice<T: Clone>(
        &mut self, 
        items: &[T], 
        shape: &[usize]
    ) -> Result<DTensor, RandomError> { ... }

    pub fn shuffle<T>(&mut self, arr: &mut DTensor<T, _>) {
        // Fisher-Yates shuffle implementation
    }
    
    /// Create a new shuffled copy without modifying the original
    pub fn permutation<T: Clone>(&mut self, arr: &mut DTensor<T, _>) -> DTensor<T, _> {
        // Clone and shuffle
    }
    
    /// Generate a random permutation of indices 0..n
    pub fn permutation_indices(&mut self, n: usize) -> Vec<usize> {
        // Generate [0,1,2,...,n-1] then shuffle
    }
}
```

## Supported Distributions

### Phase 1 (Core Distributions)
- **Uniform** (float)
- **Uniform** (int)
- **Gaussian/Normal**
- **Bernoulli**
- **Poisson**
- **Exponential**
- **Beta**
- **Gamma**
- **Multinomial**
- **Dirichlet**

### Phase 2 (Extended Distributions)
- **Multivariate Normal**
- **Cauchy**
- **Logistic**
- **Geometric**
- **Negative Binomial**
- **Chi-square**
- **Student's T**
- **Pareto**
- **Zipf**
- **Hypergeometric**
- **Von Mises** (for angles)
- **Wishart**

## Additional Features

### Shuffling Operations

Implementation using Fisher-Yates shuffle algorithm with `g.next_u64()` or `g.next_f64()` for index selection.

### Scalar Generation

Internal `scalar.rs` module with a `RandomScalar` trait for unified random number generation across types (`f32`, `f64`, `c32`, `c64`).

### Alphabet Manipulation

Support for `randint()` and `randchoice()` methods to work with character alphabets and discrete choices.

## Performance Optimizations

### Vectorization
- For large samples (Uniform/Normal), implement vectorized versions (e.g., Box-Muller in blocks)
- Enable with `feature = "simd"`
- Reference: https://github.com/lfarizav/pseudorandomnumbergenerators

### Parallelism
- Use `rayon` to divide sampling regions
- Each chunk gets its own `SmallRng` derived from the global seed (SplitMix or reseeding)
- Ensures reproducibility across parallel execution

### RNG Selection
- **Default**: `SmallRng` (fast) for general use
- **Optional**: ChaCha/PCG via features for cryptographically secure or cross-language reproducible results

## Testing Strategy

### Statistical Validation
- Cross-check statistical results (mean/variance) against `numpy.random` for basic distributions

### Property Testing
- Use `quickcheck` for invariant testing (e.g., `sum(weights) ~ 1.0`)

### Reproducibility Testing
- Verify same seed produces identical sequences

## Usage Examples

```rust
// Create a generator backend
let mut g = SmallRng::from_seed(42);

// Uniform distribution
let dist = Uniform::new(0.0, 1.0);
let x: f64 = dist.sample(&mut g);
let a = dist.sample_tensor::<f64>(&mut g, &[2, 2]);

// Normal distribution
let normal = Normal::new(0.0, 1.0);
let z = normal.sample(&mut g);

// Random integers
let uniform_int = UniformInt::new(0, 10);
let ids = uniform_int.sample_tensor(&mut g, &[10]);

// Choice from alphabet
let alphabet = vec!['a', 'b', 'c', 'd'];
let choice = RandChoice::new(&alphabet);
let word = choice.sample_tensor(&mut g, &[5]); // 5 letters

// Poisson(λ=3) with shape (100,)
let pois = Poisson::new(3.0);
let counts = pois.sample_tensor(&mut g, &[100]);

// Shuffle in-place (like numpy.random.shuffle)
let mut arr = vec![1, 2, 3, 4, 5];
g.shuffle(&mut arr);

// Create new permutation without modifying original
let arr2 = vec![10, 20, 30, 40];
let perm = g.permutation(&arr2); // e.g., [30, 10, 40, 20]

// Generate permutation of indices 0..n
let idx = g.permutation_indices(5); // e.g., [2, 0, 4, 1, 3]
```
