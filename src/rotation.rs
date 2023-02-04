use crate::autodiff::ValueOrDerivative;

pub fn cross_product<T>(x: &[T], y: &[T], x_cross_y: &mut [T])
where
    T: ValueOrDerivative,
{
    x_cross_y[0] = x[1] * y[2] - x[2] * y[1];
    x_cross_y[1] = x[2] * y[0] - x[0] * y[2];
    x_cross_y[2] = x[0] * y[1] - x[1] * y[0];
}

pub fn dot_product<T>(x: &[T], y: &[T]) -> T
where
    T: ValueOrDerivative,
{
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

pub fn angle_axis_rotate_point<T>(angle_axis: &[T], pt: &[T], result: &mut [T])
where
    T: ValueOrDerivative,
{
    let theta2 = dot_product(angle_axis, angle_axis);
    if theta2 > T::epsilon() {
        let theta = theta2.sqrt();
        let costheta = theta.cos();
        let sintheta = theta.sin();
        let theta_inverse = (T::one() / theta).into();

        let w = vec![
            angle_axis[0] * theta_inverse,
            angle_axis[1] * theta_inverse,
            angle_axis[2] * theta_inverse,
        ];

        let w_cross_pt = vec![
            w[1] * pt[2] - w[2] * pt[1],
            w[2] * pt[0] - w[0] * pt[2],
            w[0] * pt[1] - w[1] * pt[0],
        ];
        let tmp = (w[0] * pt[0] + w[1] * pt[1] + w[2] * pt[2]) * (T::one() - costheta);

        result[0] = pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
        result[1] = pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
        result[2] = pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;
    } else {
        let w_cross_pt = vec![
            angle_axis[1] * pt[2] - angle_axis[2] * pt[1],
            angle_axis[2] * pt[0] - angle_axis[0] * pt[2],
            angle_axis[0] * pt[1] - angle_axis[1] * pt[0],
        ];

        result[0] = pt[0] + w_cross_pt[0];
        result[1] = pt[1] + w_cross_pt[1];
        result[2] = pt[2] + w_cross_pt[2];
    }
}

pub fn unit_quaternion_rotate_point<T>(q: &[T], pt: &[T], result: &mut [T])
where
    T: ValueOrDerivative,
{
    let mut uv0 = q[2] * pt[2] - q[3] * pt[1];
    let mut uv1 = q[3] * pt[0] - q[1] * pt[2];
    let mut uv2 = q[1] * pt[1] - q[2] * pt[0];
    uv0 += uv0;
    uv1 += uv1;
    uv2 += uv2;
    result[0] = pt[0] + q[0] * uv0;
    result[1] = pt[1] + q[0] * uv1;
    result[2] = pt[2] + q[0] * uv2;
    result[0] += q[2] * uv2 - q[3] * uv1;
    result[1] += q[3] * uv0 - q[1] * uv2;
    result[2] += q[1] * uv1 - q[2] * uv0;
}

pub fn quaternion_rotate_point<T>(q: &[T], pt: &[T], result: &mut [T])
where
    T: ValueOrDerivative,
{
    let scale = T::one() / (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();

    let unit = [scale * q[0], scale * q[1], scale * q[2], scale * q[3]];

    unit_quaternion_rotate_point(&unit, pt, result);
}
