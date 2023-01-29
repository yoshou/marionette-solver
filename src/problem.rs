use crate::autodiff::{Dual, Functor};
use crate::sparse_matrix::{CsrBlockMatrix};

extern crate nalgebra as na;

#[derive(Debug, Clone, Copy)]
pub struct ParameterBlock {
    pub offset: usize,
    pub size: usize,
}

impl ParameterBlock {
    pub fn new(offset: usize, size: usize) -> Self {
        ParameterBlock {
            offset: offset,
            size: size,
        }
    }
}

pub trait ResidualVec {
    fn eval(&self, params: &Vec<f64>, values: &mut Vec<f64>) -> bool;
    fn jacobian(&self, params: &Vec<f64>, jacob: &mut Vec<na::DMatrix<f64>>)
        -> bool;

    fn num_residuals(&self) -> usize;
    fn parameters(&self) -> &Vec<ParameterBlock>;
}

pub struct AutoDiffResidualVec<T: Functor> {
    f: T,
    params: Vec<ParameterBlock>,
}

impl<T: Functor> AutoDiffResidualVec<T> {
    pub fn new(f: T, params: Vec<ParameterBlock>) -> Self {
        AutoDiffResidualVec {
            f: f,
            params: params,
        }
    }
}

impl<T: Functor> ResidualVec for AutoDiffResidualVec<T> {
    fn eval(&self, params: &Vec<f64>, values: &mut Vec<f64>) -> bool {
        let param_blocks = self
            .params
            .iter()
            .map(|x| params[x.offset..(x.offset + x.size)].to_vec())
            .collect();
        self.f.invoke(&param_blocks, values)
    }

    fn parameters(&self) -> &Vec<ParameterBlock> {
        &self.params
    }

    fn jacobian(
        &self,
        params: &Vec<f64>,
        jacob: &mut Vec<na::DMatrix<f64>>,
    ) -> bool {
        let mut v = params
            .iter()
            .map(|x| Dual { a: *x, b: 0.0 })
            .collect::<Vec<Dual>>();

        for block in &self.params {
            let mut jacob_columns = Vec::<na::DVector<f64>>::new();
            for i in block.offset..(block.offset + block.size) {
                v[i].b = 1.0;

                let mut residuals = vec![Dual::default(); self.f.num_residuals()];
                let param_blocks = self
                    .params
                    .iter()
                    .map(|x| v[x.offset..(x.offset + x.size)].to_vec())
                    .collect();

                if !self.f.invoke(&param_blocks, &mut residuals) {
                    return false;
                }

                jacob_columns.push(na::DVector::from_vec(residuals.iter().map(|x| x.b).collect()));

                v[i].b = 0.0;
            }

            jacob.push(na::DMatrix::from_columns(&jacob_columns));
        }

        true
    }

    fn num_residuals(&self) -> usize {
        self.f.num_residuals()
    }
}
pub struct NonlinearLeastSquaresProblem {
    pub residuals: Vec<Box<dyn ResidualVec>>,
}

impl NonlinearLeastSquaresProblem {
    pub fn new() -> Self {
        NonlinearLeastSquaresProblem {
            residuals: Vec::<Box<dyn ResidualVec>>::new(),
        }
    }
}

enum TerminationStatus {
    NotConverged,
    Converged,
}

pub struct SolveResult {
    termination_status: TerminationStatus,
}

pub trait NonlinearLeastSquaresSolver {
    fn solve(&mut self, p: &NonlinearLeastSquaresProblem) -> SolveResult;
}

pub struct TrustRegionSolver {
    pub params: Vec<f64>,
    iteration: TrustRegionSolverIteration,
    max_iteration: u32,
    gradient_tolerance: f64,
    mu: f64,
    nu: f64,
    eta: f64,
    max_mu: f64,
    function_tolerance: f64,
}

impl TrustRegionSolver {
    pub fn new(params: Vec<f64>) -> Self {
        TrustRegionSolver {
            params: params,
            iteration: TrustRegionSolverIteration {
                num: 0,
                gradient_max_norm: 1e+10,
                gradient_norm: 1e+10,
            },
            max_iteration: 10,
            gradient_tolerance: 1e-10,
            mu: 10000.0,
            nu: 2.0,
            eta: 0.001,
            max_mu: 10000000000000000.0,
            function_tolerance: 1.0e-6,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct TrustRegionSolverIteration {
    num: u32,
    gradient_max_norm: f64,
    gradient_norm: f64,
}

fn solve_upper_triangular(
    q: &na::DMatrix<f64>,
    r: &na::DMatrix<f64>,
    b: &na::DVector<f64>,
    x: &mut na::DVector<f64>,
) -> bool {
    let dim = q.ncols();

    let qt_mul_b = q.transpose() * b;

    for i in (0..dim).rev() {
        let diag = r[(i, i)];

        x[i] = (qt_mul_b[i]
            - r.slice_range(i..(i + 1), (i + 1)..)
                .dot(&x.transpose().slice_range(0..1, (i + 1)..)))
            / diag;
    }

    true
}

impl TrustRegionSolver {
    fn evaluate_grad_and_jacobian(
        &self,
        p: &NonlinearLeastSquaresProblem,
    ) -> Option<(na::DVector<f64>, na::DVector<f64>, na::DMatrix<f64>)> {
        let sum_num_residuals = p.residuals.iter().map(|x| x.num_residuals()).sum();
        let mut val = na::DVector::<f64>::zeros(sum_num_residuals);

        let mut j = CsrBlockMatrix::<f64>::new();

        let mut row_pos = 0;
        for residual in &p.residuals {
            let mut residual_jacob = Vec::<na::DMatrix<f64>>::new();
            let mut residual_val = vec![0.0; residual.num_residuals()];

            if !residual.jacobian(&self.params, &mut residual_jacob) {
                return None;
            }

            j.add_row(residual.num_residuals());

            let mut it = residual.parameters().iter().zip(&residual_jacob).collect::<Vec<_>>();
            it.sort_by_key(|(a, b)| a.offset);

            for (param_block, jacob_block) in it {
                if !j.add_row_block(param_block.offset, &jacob_block) {
                    return None;
                }
            }

            if !residual.eval(&self.params, &mut residual_val) {
                return None;
            }

            val.slice_mut((row_pos, 0), (residual.num_residuals(), 1))
                .copy_from_slice(&residual_val);

            row_pos = row_pos + residual.num_residuals()
        }

        let jac = j.to_dense_matrix();

        let grad = jac.transpose() * &val;

        Some((val, grad, jac))
    }

    fn compute_step(&self, val: &na::DVector<f64>, jac: &na::DMatrix<f64>) -> na::DVector<f64> {
        let jac_scale = (jac.transpose() * jac)
            .diagonal()
            .map(|x| 1.0 / (1.0 + x.sqrt()));

        let jac_scaled = jac * na::DMatrix::from_diagonal(&jac_scale);

        let lambda = 1.0 / self.mu;
        let diag = ((jac_scaled.transpose() * &jac_scaled).map(|x| x.clamp(1.0e-6, 1.0e+32))
            * lambda)
            .map(|x| x.sqrt())
            .diagonal();

        let mut lhs =
            na::DMatrix::<f64>::zeros(jac_scaled.nrows() + diag.nrows(), jac_scaled.ncols());
        let mut rhs = na::DVector::<f64>::zeros(jac_scaled.nrows() + diag.nrows());

        lhs.slice_mut((0, 0), (jac_scaled.nrows(), jac_scaled.ncols()))
            .copy_from(&jac_scaled);
        lhs.slice_mut((jac_scaled.nrows(), 0), (diag.nrows(), jac_scaled.ncols()))
            .copy_from(&na::DMatrix::from_diagonal(&diag));

        rhs.slice_mut((0, 0), (val.nrows(), 1)).copy_from(&val);

        let qr = &lhs.qr();

        let q = qr.q();
        let r = qr.r();

        let mut x = na::DVector::<f64>::zeros(q.ncols());
        solve_upper_triangular(&q, &r, &rhs, &mut x);

        -(na::DMatrix::from_diagonal(&jac_scale) * x)
    }

    fn need_next_iteration(&self) -> bool {
        if self.iteration.num >= self.max_iteration {
            false
        } else if self.iteration.gradient_max_norm < self.gradient_tolerance {
            false
        } else {
            true
        }
    }

    fn eval_objective_function(
        &self,
        p: &NonlinearLeastSquaresProblem,
        params: &na::DVector<f64>,
    ) -> f64 {
        let sum_num_residuals = p.residuals.iter().map(|x| x.num_residuals()).sum();
        let mut val = na::DVector::<f64>::zeros(sum_num_residuals);

        let mut row_pos = 0;
        for residual in &p.residuals {
            let mut residual_val = vec![0.0; residual.num_residuals()];

            residual.eval(&params.data.as_vec(), &mut residual_val);

            val.slice_mut((row_pos, 0), (residual.num_residuals(), 1))
                .copy_from_slice(&residual_val);

            row_pos = row_pos + residual.num_residuals()
        }

        val.norm_squared() / 2.0
    }

    fn eval_model_function(
        &self,
        jac: &na::DMatrix<f64>,
        val: &na::DVector<f64>,
        step: &na::DVector<f64>,
    ) -> f64 {
        (jac * step + val).norm_squared() / 2.0
    }
}

impl NonlinearLeastSquaresSolver for TrustRegionSolver {
    fn solve(&mut self, p: &NonlinearLeastSquaresProblem) -> SolveResult {
        let mut result = SolveResult {
            termination_status: TerminationStatus::NotConverged,
        };

        while self.need_next_iteration() {
            if let Some((val, grad, jac)) = self.evaluate_grad_and_jacobian(p) {
                self.iteration.gradient_max_norm = grad.abs().max();
                self.iteration.gradient_norm = grad.norm();

                let step = self.compute_step(&val, &jac);

                let m_p = self.eval_model_function(&jac, &val, &step);
                let m_0 = self.eval_model_function(&jac, &val, &na::DVector::zeros(step.nrows()));

                let x = na::DVector::from_vec(self.params.clone());

                let f_x_plus_step = self.eval_objective_function(p, &(&x + &step));
                let f_x = self.eval_objective_function(p, &x);

                let cost_change = f_x_plus_step - f_x;

                if cost_change.abs() / f_x <= self.function_tolerance {
                    result.termination_status = TerminationStatus::Converged;
                    break;
                }

                let model_change = m_p - m_0;

                let rho = cost_change / model_change;

                if rho > self.eta {
                    self.params = (&x + &step).data.as_vec().clone();
                    self.mu = f64::min(
                        self.mu / f64::max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0).powi(3)),
                        self.max_mu,
                    );
                    self.nu = 2.0;
                } else {
                    self.mu = self.mu / self.nu;
                    self.nu = self.nu * 2.0;
                }

                self.iteration.num = self.iteration.num + 1;
            } else {
                return result;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use crate::problem::solve_upper_triangular;

    macro_rules! assert_delta {
        ($x:expr, $y:expr, $d:expr) => {
            if !($x - $y < $d || $y - $x < $d) {
                panic!();
            }
        };
    }

    #[test]
    fn solve_upper_triangular_test() {
        let a = nalgebra::DMatrix::from_row_slice(
            4,
            4,
            &[
                2.0, 3.0, 6.0, -2.0, 1.0, 4.0, -2.0, 4.0, 4.0, 1.0, 7.0, -5.0, 3.0, 7.0, 3.0, 2.0,
            ],
        );
        let b = nalgebra::DVector::from_row_slice(&[5.0, 5.0, 8.0, 9.0]);

        let qr = a.qr();

        let mut x = nalgebra::DVector::zeros(b.nrows());

        solve_upper_triangular(&qr.q(), &qr.r(), &b, &mut x);

        assert_delta!(x[0], 3.0, 1.0e-7);
        assert_delta!(x[1], -1.0, 1.0e-7);
        assert_delta!(x[2], 1.0, 1.0e-7);
        assert_delta!(x[3], 2.0, 1.0e-7);
    }
}
