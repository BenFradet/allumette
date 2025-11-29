use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait Element:
    Clone
    + Copy
    + std::fmt::Debug
    + PartialEq
    + PartialOrd
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Neg<Output = Self>
where
    Self: Sized,
{
    fn zero() -> Self;
    fn one() -> Self;
    fn two() -> Self;
    fn fromf(f: f64) -> Self;
    fn is_close(self, rhs: Self) -> bool;

    fn powf(self, exp: Self) -> Self;

    fn exp(self) -> Self;

    fn ln(self) -> Self;

    fn sig(self) -> Self {
        if self >= Self::zero() {
            Self::one() / (Self::one() + (-self).exp())
        } else {
            let exp = self.exp();
            exp / (Self::one() + exp)
        }
    }

    fn relu(self) -> Self {
        if self > Self::zero() {
            self
        } else {
            Self::zero()
        }
    }

    fn relu_diff(self, d: Self) -> Self {
        if self > Self::zero() {
            d
        } else {
            Self::zero()
        }
    }
}

impl Element for f32 {
    fn one() -> Self {
        1.
    }

    fn zero() -> Self {
        0.
    }

    fn two() -> Self {
        2.
    }

    fn fromf(f: f64) -> Self {
        f as f32
    }

    // gpu is less precise
    fn is_close(self, rhs: Self) -> bool {
        (self - rhs).abs() < 1e-2
    }

    fn exp(self) -> Self {
        self.exp()
    }

    fn ln(self) -> Self {
        self.ln()
    }

    fn powf(self, exp: Self) -> Self {
        self.powf(exp)
    }
}

impl Element for f64 {
    fn one() -> Self {
        1.
    }

    fn zero() -> Self {
        0.
    }

    fn two() -> Self {
        2.
    }

    fn fromf(f: f64) -> Self {
        f
    }

    fn is_close(self, rhs: Self) -> bool {
        (self - rhs).abs() < 1e-4
    }

    fn exp(self) -> Self {
        self.exp()
    }

    fn ln(self) -> Self {
        self.ln()
    }

    fn powf(self, exp: Self) -> Self {
        self.powf(exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn sig_tests(a in any::<f64>()) {
            let f = a.sig();
            if a >= 0. {
                assert_eq!(1. / (1. + (-a).exp()), f);
            } else {
                assert_eq!(a.exp() / (1. + a.exp()), f);
            }
        }

        #[test]
        fn relu_tests(a in any::<f64>()) {
            let f = a.relu();
            if a >= 0. {
                assert_eq!(a, f);
            } else {
                assert_eq!(0., f);
            }
            let back = a.relu_diff(a);
            if a > 0. {
                assert_eq!(a, back);
            } else {
                assert_eq!(0., back);
            }
        }
    }
}
