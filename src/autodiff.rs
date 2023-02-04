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
    fn epsilon() -> Self;
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
    fn epsilon() -> Self {
        Dual {
            a: f64::EPSILON,
            b: 0.0,
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
    fn epsilon() -> Self {
        f64::EPSILON
    }
}

pub trait Functor {
    fn invoke<T>(&self, params: &Vec<Vec<T>>, residuals: &mut Vec<T>) -> bool
    where
        T: ValueOrDerivative + Default;
    fn num_residuals(&self) -> usize;
}
