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

    fn inv_back(self, d: Self) -> Self {
        if self == Self::zero() {
            -d
        } else {
            -d / (self.powf(Self::two()))
        }
    }

    fn exp(self) -> Self;
    fn exp_back(self, d: Self) -> Self {
        self.exp() * d
    }

    fn ln(self) -> Self;
    fn ln_back(self, d: Self) -> Self {
        if self == Self::zero() {
            d
        } else {
            d / self
        }
    }

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

    fn relu_back(self, d: Self) -> Self {
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
        fn ln_tests(a in any::<f64>()) {
            let f = if a > 0. { a.ln() } else { 0. };
            if a <= 0. {
                assert_eq!(0., f);
            } else {
                assert_eq!(a.ln(), f);
            }
            let back = a.ln_back(a);
            let exp = if a == 0. { 1. } else { a };
            assert_eq!(a / exp, back);
        }

        #[test]
        fn inv_tests(a in any::<f64>()) {
            let f = if a != 0. { 1. / a } else { 0. };
            if a == 0. {
                assert_eq!(0., f);
            } else {
                assert_eq!(1. / a, f);
            }
            let back = a.inv_back(a);
            let exp = if a == 0. { 1. } else { a };
            assert_eq!(-a / (exp.powf(2.)), back);
        }

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
            let back = a.relu_back(a);
            if a > 0. {
                assert_eq!(a, back);
            } else {
                assert_eq!(0., back);
            }
        }

        #[test]
        fn exp_tests(a in any::<f64>()) {
            let f = a.exp();
            assert_eq!(a.exp(), f);
            let back = a.exp_back(a);
            assert_eq!(a.exp() * a, back);
        }
    }
}
