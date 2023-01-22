# marionette-solver

Optimization library for pose graph
 
## Usage samples

#### Basic usage
```rust
fn main() {
    let params = vec![0.0; num_params];

    let mut solver = TrustRegionSolver::new(params);
    solver.solve(&prob);
}
```