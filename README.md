# marionette-solver

Optimization library for pose graph
 
## Usage samples

#### Basic usage
```rust
fn main() {
    let params = vec![0.0; num_params];
    let method = Box::new(LevenbergMarquardtMethod::new(Box::new(
        LevenbergMarquardtDenseSchurComplementSolver::new(Box::new(
            BundleAdjustmentProblemStructure {},
        )),
    )));

    let mut solver = TrustRegionSolver::new(params, method);
    solver.solve(&prob);
}
```