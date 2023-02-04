extern crate nalgebra as na;

use crate::nls_problem::TrustRegionMethod;
use crate::sparse_matrix::CsrBlockMatrix;
use nalgebra_sparse::factorization::CscCholesky;

pub trait LevenbergMarquardtLinearSolver {
    fn solve(
        &self,
        jacobian: &CsrBlockMatrix<f64>,
        diag: &na::DVector<f64>,
        residuals: &na::DVector<f64>,
        x: &mut na::DVector<f64>,
    );
}

pub struct LevenbergMarquardtDenseQRSolver {}

impl LevenbergMarquardtDenseQRSolver {
    pub fn new() -> Self {
        LevenbergMarquardtDenseQRSolver {}
    }
}

impl LevenbergMarquardtLinearSolver for LevenbergMarquardtDenseQRSolver {
    fn solve(
        &self,
        jacobian: &CsrBlockMatrix<f64>,
        diag: &na::DVector<f64>,
        residuals: &na::DVector<f64>,
        x: &mut na::DVector<f64>,
    ) {
        let jac = jacobian.to_dense_matrix();

        let mut lhs = na::DMatrix::<f64>::zeros(jac.nrows() + diag.nrows(), jac.ncols());
        let mut rhs = na::DVector::<f64>::zeros(jac.nrows() + diag.nrows());

        lhs.view_mut((0, 0), (jac.nrows(), jac.ncols()))
            .copy_from(&jac);
        lhs.view_mut((jac.nrows(), 0), (diag.nrows(), jac.ncols()))
            .copy_from(&na::DMatrix::from_diagonal(&diag));

        rhs.rows_range_mut(..residuals.nrows())
            .copy_from(&residuals);

        let qr = &lhs.qr();

        let q = qr.q();
        let r = qr.r();

        solve_upper_triangular(&q, &r, &rhs, x);
    }
}

pub struct LevenbergMarquardtDenseNormalCholeskySolver {}

impl LevenbergMarquardtDenseNormalCholeskySolver {
    pub fn new() -> Self {
        LevenbergMarquardtDenseNormalCholeskySolver {}
    }
}

impl LevenbergMarquardtLinearSolver for LevenbergMarquardtDenseNormalCholeskySolver {
    fn solve(
        &self,
        jacobian: &CsrBlockMatrix<f64>,
        diag: &na::DVector<f64>,
        residuals: &na::DVector<f64>,
        x: &mut na::DVector<f64>,
    ) {
        let jac = jacobian.to_dense_matrix();

        let lhs = jac.transpose() * &jac + na::DMatrix::from_diagonal(&diag.map(|x| x * x));
        let rhs = jac.transpose() * residuals;

        let cholesky = &lhs.cholesky().unwrap();

        x.copy_from(&cholesky.solve(&rhs));
    }
}

pub struct LevenbergMarquardtSparseNormalCholeskySolver {}

impl LevenbergMarquardtSparseNormalCholeskySolver {
    pub fn new() -> Self {
        LevenbergMarquardtSparseNormalCholeskySolver {}
    }
}

impl LevenbergMarquardtLinearSolver for LevenbergMarquardtSparseNormalCholeskySolver {
    fn solve(
        &self,
        jacobian: &CsrBlockMatrix<f64>,
        diag: &na::DVector<f64>,
        residuals: &na::DVector<f64>,
        x: &mut na::DVector<f64>,
    ) {
        use nalgebra_sparse::{coo::CooMatrix, csc::CscMatrix};

        let jac = CscMatrix::from(&jacobian.to_sparse_matrix());

        let mut diag_coo = CooMatrix::<f64>::new(diag.nrows(), diag.nrows());
        for i in 0..diag.nrows() {
            diag_coo.push(i, i, diag[i] * diag[i]);
        }

        let lhs = jac.transpose() * &jac + CscMatrix::from(&diag_coo);
        let rhs = jacobian.transpose_and_mul(&residuals);

        let cholesky = CscCholesky::factor(&lhs).unwrap();

        x.copy_from(&cholesky.solve(&rhs));
    }
}

pub struct LevenbergMarquardtDenseSchurComplementSolver {
    structure: Box<dyn SchurStructure>,
}

impl LevenbergMarquardtDenseSchurComplementSolver {
    pub fn new(structure: Box<dyn SchurStructure>) -> Self {
        LevenbergMarquardtDenseSchurComplementSolver {
            structure: structure,
        }
    }
}

pub trait SchurStructure {
    fn reduced_size(&self, jacobian: &CsrBlockMatrix<f64>) -> Option<usize>;
}

pub struct BundleAdjustmentProblemStructure {}

impl SchurStructure for BundleAdjustmentProblemStructure {
    fn reduced_size(&self, jacobian: &CsrBlockMatrix<f64>) -> Option<usize> {
        let mut s = 0;
        let mut c = jacobian.ncols();
        for row_data in &jacobian.rows {
            // Point positions
            if let Some(column_data) = row_data.columns.first() {
                let j = column_data.column;
                let block_size = column_data.data.ncols();
                s = s.max(j + block_size);
            }

            // Camera parameters
            if row_data.columns.len() > 1 {
                for k in 1..row_data.columns.len() {
                    let column_data = &row_data.columns[k];
                    let j = column_data.column;
                    c = c.min(j);
                }
            }

            if c < s {
                return None;
            }
        }

        Some(s)
    }
}

impl LevenbergMarquardtLinearSolver for LevenbergMarquardtDenseSchurComplementSolver {
    fn solve(
        &self,
        jacobian: &CsrBlockMatrix<f64>,
        diag: &na::DVector<f64>,
        residuals: &na::DVector<f64>,
        x: &mut na::DVector<f64>,
    ) {
        use nalgebra_sparse::{coo::CooMatrix, csc::CscMatrix};
        use std::collections::HashMap;

        let reduced_size = self.structure.reduced_size(&jacobian).unwrap();
        let jac_coo = jacobian.to_sparse_matrix();

        let e_size = reduced_size;
        let f_size = jacobian.ncols() - reduced_size;

        // Block matrix:
        // J := | E F |
        //
        // M := | A B | = | (E^T * E) (E^T * F) |
        //      | C D |   | (F^T * E) (F^T * F) |
        //
        // Schur complement:
        // M/A := (E^T * F)^T * (E^T * E)^-1 * (E^T * F)

        // Compute (E^T * E)^-1
        let mut et_e_diag_inv_values = HashMap::<usize, na::DMatrix<f64>>::new();

        for row_data in &jacobian.rows {
            let column_data = row_data.columns.first().unwrap();
            let j = column_data.column;
            let block = &column_data.data;

            et_e_diag_inv_values
                .entry(j)
                .or_insert(na::DMatrix::from_diagonal(
                    &diag.rows(j, block.ncols()).map(|x| x * x),
                ));
        }

        for row_data in &jacobian.rows {
            for column_data in &row_data.columns {
                let j = column_data.column;
                let block = &column_data.data;
                if j < reduced_size {
                    let et_e_diag_inv_value = et_e_diag_inv_values.get_mut(&j).unwrap();
                    *et_e_diag_inv_value = et_e_diag_inv_value.clone() + block.transpose() * block;
                }
            }
        }

        let mut et_e_inv_coo = CooMatrix::<f64>::new(e_size, e_size);

        for m in et_e_diag_inv_values.values_mut() {
            *m = m.clone().try_inverse().unwrap();
        }

        for (j, m) in &et_e_diag_inv_values {
            et_e_inv_coo.push_matrix(*j, *j, m);
        }

        let et_e_inv = CscMatrix::from(&et_e_inv_coo);

        // Compute (F^T * F), (E^T * F)^T = (F^T * E)
        let mut e_coo = CooMatrix::<f64>::new(jacobian.nrows() + diag.nrows(), e_size);
        let mut f_coo = CooMatrix::<f64>::new(jacobian.nrows() + diag.nrows(), f_size);
        for (i, j, value) in jac_coo.triplet_iter() {
            if j < reduced_size {
                e_coo.push(i, j, *value);
            } else {
                f_coo.push(i, j - reduced_size, *value);
            }
        }
        for i in 0..diag.nrows() {
            if i < reduced_size {
                e_coo.push(jacobian.nrows() + i, i, diag[i]);
            } else {
                f_coo.push(jacobian.nrows() + i, i - reduced_size, diag[i]);
            }
        }

        let e = CscMatrix::from(&e_coo);
        let f = CscMatrix::from(&f_coo);

        let ft = f.transpose();
        let ft_f = &ft * &f;
        let ft_e = &ft * &e;

        // Compute schur components
        let lhs = na::DMatrix::from(&(&ft_f - &ft_e * &et_e_inv * ft_e.transpose()));

        let eps = jacobian.transpose_and_mul(&residuals);
        let rhs =
            eps.rows_range(reduced_size..) - &ft_e * &et_e_inv * eps.rows_range(..reduced_size);

        // Dense cholesky factorization
        let cholesky = &lhs.cholesky().unwrap();

        x.rows_range_mut(reduced_size..)
            .copy_from(&cholesky.solve(&rhs));

        // Solve backward substitution
        let y = eps.rows_range(..reduced_size) - ft_e.transpose() * x.rows_range(reduced_size..);
        for (j, m) in &et_e_diag_inv_values {
            x.rows_mut(*j, m.nrows())
                .copy_from(&(m * y.rows(*j, m.nrows())));
        }
    }
}

pub struct LevenbergMarquardtMethod {
    linear_solver: Box<dyn LevenbergMarquardtLinearSolver>,
}

impl LevenbergMarquardtMethod {
    pub fn new(linear_solver: Box<dyn LevenbergMarquardtLinearSolver>) -> Self {
        LevenbergMarquardtMethod {
            linear_solver: linear_solver,
        }
    }
}

impl TrustRegionMethod for LevenbergMarquardtMethod {
    fn compute_step(
        &self,
        val: &na::DVector<f64>,
        jacobian: &CsrBlockMatrix<f64>,
        mu: f64,
    ) -> na::DVector<f64> {
        let jac_scale = jacobian
            .column_norm_squared()
            .map(|x| 1.0 / (1.0 + x.sqrt()));

        let jac_scaled = jacobian.scale_columns(&jac_scale);

        let lambda = 1.0 / mu;
        let diag = jac_scaled
            .column_norm_squared()
            .map(|x| (x.clamp(1.0e-6, 1.0e+32) * lambda).sqrt());

        let mut x = na::DVector::<f64>::zeros(jacobian.ncols());
        self.linear_solver.solve(&jac_scaled, &diag, &val, &mut x);

        -x.component_mul(&jac_scale)
    }
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
            - r.view_range(i..(i + 1), (i + 1)..)
                .dot(&x.transpose().view_range(0..1, (i + 1)..)))
            / diag;
    }

    true
}

#[cfg(test)]
mod tests {
    use crate::levenberg_marquardt::solve_upper_triangular;

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
