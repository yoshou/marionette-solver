extern crate lm_optimize;

use serde::Deserialize;
use serde_json::Value;
use std::fs::OpenOptions;
use std::io::BufReader;

use lm_optimize::autodiff::{angle_axis_rotate_point, Functor, ValueOrDerivative};
use lm_optimize::levenberg_marquardt::{
    BundleAdjustmentProblemStructure, LevenbergMarquardtDenseSchurComplementSolver,
    LevenbergMarquardtMethod,
};
use lm_optimize::problem::{
    AutoDiffResidualVec, NonlinearLeastSquaresProblem, NonlinearLeastSquaresSolver, ParameterBlock,
    TrustRegionSolver,
};
use std::ops::AddAssign;

struct SnavelyReprojectionError {
    observed_x: f64,
    observed_y: f64,
}

impl SnavelyReprojectionError {
    fn new(observed_x: f64, observed_y: f64) -> Self {
        SnavelyReprojectionError {
            observed_x: observed_x,
            observed_y: observed_y,
        }
    }
}

impl Functor for SnavelyReprojectionError {
    fn num_residuals(&self) -> usize {
        2
    }
    fn invoke<T>(&self, params: &Vec<Vec<T>>, residuals: &mut Vec<T>) -> bool
    where
        T: ValueOrDerivative + Default + AddAssign,
    {
        let camera = &params[0];
        let point = &params[1];

        let mut p = [T::default(); 3];
        angle_axis_rotate_point(&camera[0..3], &point[0..3], &mut p);
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        let xp = -p[0] / p[2];
        let yp = -p[1] / p[2];

        let l1 = camera[7];
        let l2 = camera[8];
        let r2 = xp * xp + yp * yp;
        let distortion = T::one() + r2 * (l1 + l2 * r2);
        let focal = camera[6];
        let predicted_x = focal * distortion * xp;
        let predicted_y = focal * distortion * yp;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T::from(self.observed_x).unwrap();
        residuals[1] = predicted_y - T::from(self.observed_y).unwrap();

        true
    }
}

fn main() {
    let prob_file = OpenOptions::new()
        .read(true)
        .write(false)
        .open("examples/ba/ba_problem.json")
        .unwrap();
    let reader = BufReader::new(prob_file);

    let v: Value = serde_json::from_reader(reader).unwrap();

    let camera_idxs = Vec::<usize>::deserialize(&v["camera_idxs"]).unwrap();
    let point_idxs = Vec::<usize>::deserialize(&v["point_idxs"]).unwrap();
    let observations = Vec::<[f64; 2]>::deserialize(&v["observations"]).unwrap();
    let points = Vec::<[f64; 3]>::deserialize(&v["points"]).unwrap();
    let cameras = Vec::<[f64; 9]>::deserialize(&v["cameras"]).unwrap();

    let point_params_offset = 0;
    let camera_params_offset = point_params_offset + 3 * points.len();
    let num_params = 9 * cameras.len() + 3 * points.len();

    let camera_param_blocks = (0..cameras.len())
        .map(|x| ParameterBlock::new(camera_params_offset + x * 9, 9))
        .collect::<Vec<ParameterBlock>>();

    let point_param_blocks = (0..points.len())
        .map(|x| ParameterBlock::new(point_params_offset + x * 3, 3))
        .collect::<Vec<ParameterBlock>>();

    let mut prob = NonlinearLeastSquaresProblem::new();

    for i in 0..observations.len() {
        let functor = SnavelyReprojectionError::new(observations[i][0], observations[i][1]);

        prob.residuals.push(Box::new(AutoDiffResidualVec::new(
            functor,
            vec![
                camera_param_blocks[camera_idxs[i]],
                point_param_blocks[point_idxs[i]],
            ],
        )));
    }

    let mut params = vec![0.0; num_params];
    for i in 0..cameras.len() {
        for j in 0..9 {
            params[camera_params_offset + i * 9 + j] = cameras[i][j];
        }
    }
    for i in 0..points.len() {
        for j in 0..3 {
            params[point_params_offset + i * 3 + j] = points[i][j];
        }
    }
    let method = Box::new(LevenbergMarquardtMethod::new(Box::new(
        LevenbergMarquardtDenseSchurComplementSolver::new(Box::new(
            BundleAdjustmentProblemStructure {},
        )),
    )));

    let mut solver = TrustRegionSolver::new(params, method);
    solver.solve(&prob);
}
