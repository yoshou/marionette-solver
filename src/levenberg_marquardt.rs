extern crate nalgebra as na;

use crate::sparse_matrix::CsrBlockMatrix;

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

        lhs.slice_mut((0, 0), (jac.nrows(), jac.ncols()))
            .copy_from(&jac);
        lhs.slice_mut((jac.nrows(), 0), (diag.nrows(), jac.ncols()))
            .copy_from(&na::DMatrix::from_diagonal(&diag));

        rhs.slice_mut((0, 0), (residuals.nrows(), 1))
            .copy_from(&residuals);

        let qr = &lhs.qr();

        let q = qr.q();
        let r = qr.r();

        solve_upper_triangular(&q, &r, &rhs, x);
    }
}

pub struct LevenbergMarquardtDenseNormalCholeskySolver {}

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
