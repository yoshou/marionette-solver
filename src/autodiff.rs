use num_traits::{NumCast, One, ToPrimitive, Zero};

use std::cmp::Ordering;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, Default)]
pub struct Dual {
    pub a: f64,
    pub b: f64,
}

pub trait ValueOrDerivative:
    Zero
    + One
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + AddAssign
    + NumCast
    + Copy
    + PartialEq
    + PartialOrd
    + Neg<Output = Self>
{
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
}

impl Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Dual {
            a: -self.a,
            b: -self.b,
        }
    }
}

impl Add for Dual {
    type Output = Self;
    fn add(self, rhs: Dual) -> Self::Output {
        Dual {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }
}
impl Sub for Dual {
    type Output = Self;
    fn sub(self, rhs: Dual) -> Self::Output {
        Dual {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }
}
impl Mul for Dual {
    type Output = Self;
    fn mul(self, rhs: Dual) -> Self::Output {
        Dual {
            a: self.a * rhs.a,
            b: self.a * rhs.b + self.b * rhs.a,
        }
    }
}
impl Div for Dual {
    type Output = Self;
    fn div(self, rhs: Dual) -> Self::Output {
        Dual {
            a: self.a / rhs.a,
            b: self.b / rhs.a - self.a * rhs.b / (rhs.a * rhs.a),
        }
    }
}
impl AddAssign for Dual {
    fn add_assign(&mut self, rhs: Dual) {
        *self = *self + rhs;
    }
}

impl PartialEq for Dual {
    fn eq(&self, other: &Dual) -> bool {
        self.a == other.a && self.b == other.b
    }
}

impl Zero for Dual {
    fn zero() -> Self {
        Dual { a: 0.0, b: 0.0 }
    }

    fn is_zero(&self) -> bool {
        self.a.is_zero()
    }
}

impl One for Dual {
    fn one() -> Self {
        Dual { a: 1.0, b: 0.0 }
    }
}

impl PartialOrd for Dual {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.a.partial_cmp(&other.a)
    }
}

impl NumCast for Dual {
    fn from<N: ToPrimitive>(n: N) -> Option<Dual> {
        if let Some(value) = n.to_f64() {
            Some(Dual { a: value, b: 0.0 })
        } else {
            None
        }
    }
}

impl ToPrimitive for Dual {
    fn to_i64(&self) -> Option<i64> {
        self.a.to_i64()
    }
    fn to_u64(&self) -> Option<u64> {
        self.a.to_u64()
    }
}

impl ValueOrDerivative for Dual {
    fn sqrt(self) -> Self {
        let df_0 = self.a.sqrt();
        let df_1 = 1.0 / (2.0 * df_0);
        Dual {
            a: df_0,
            b: self.b * df_1,
        }
    }
    fn sin(self) -> Self {
        let df_0 = self.a.sin();
        let df_1 = self.a.cos();
        Dual {
            a: df_0,
            b: self.b * df_1,
        }
    }
    fn cos(self) -> Self {
        let df_0 = self.a.cos();
        let df_1 = -self.a.sin();
        Dual {
            a: df_0,
            b: self.b * df_1,
        }
    }
}

impl ValueOrDerivative for f64 {
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn sin(self) -> Self {
        self.sin()
    }
    fn cos(self) -> Self {
        self.cos()
    }
}

pub trait Functor {
    fn invoke<T>(&self, params: &Vec<Vec<T>>, residuals: &mut Vec<T>) -> bool
    where
        T: ValueOrDerivative + Default;
    fn num_residuals(&self) -> usize;
}

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
    if theta2 > T::from(std::f32::EPSILON).unwrap() {
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
