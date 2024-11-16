use std::ops;

use rand::{thread_rng, Rng};

use crate::{
    autodiff::forward::Forward,
    ops::{
        binary_ops::{Add, Div, Eq, Lt, Mul},
        unary_ops::{Exp, Ln, Neg, Relu, Sig},
    },
};

use super::scalar_history::ScalarHistory;

// TODO: abstract over f64
#[derive(Debug)]
pub struct Scalar {
    pub v: f64,
    derivative: Option<f64>,
    pub history: ScalarHistory,
    id: u64,
}

impl Scalar {
    pub fn new(v: f64) -> Self {
        let mut rng = thread_rng();
        Self {
            v,
            derivative: None,
            history: ScalarHistory::default(),
            id: rng.gen(),
        }
    }

    pub fn history(mut self, history: ScalarHistory) -> Self {
        self.history = history;
        self
    }

    pub fn ln(self) -> Self {
        Forward::unary(Ln {}, self)
    }

    pub fn exp(self) -> Self {
        Forward::unary(Exp {}, self)
    }

    pub fn sig(self) -> Self {
        Forward::unary(Sig {}, self)
    }

    pub fn relu(self) -> Self {
        Forward::unary(Relu {}, self)
    }

    pub fn eq(self, rhs: Scalar) -> Self {
        Forward::binary(Eq {}, self, rhs)
    }

    pub fn lt(self, rhs: Scalar) -> Self {
        Forward::binary(Lt {}, self, rhs)
    }

    pub fn gt(self, rhs: Scalar) -> Self {
        Forward::binary(Lt {}, rhs, self)
    }
}

impl ops::Add<Scalar> for Scalar {
    type Output = Scalar;

    fn add(self, rhs: Scalar) -> Self::Output {
        Forward::binary(Add {}, self, rhs)
    }
}

impl ops::Mul<Scalar> for Scalar {
    type Output = Scalar;

    fn mul(self, rhs: Scalar) -> Self::Output {
        Forward::binary(Mul {}, self, rhs)
    }
}

impl ops::Div<Scalar> for Scalar {
    type Output = Scalar;

    fn div(self, rhs: Scalar) -> Self::Output {
        Forward::binary(Div {}, self, rhs)
    }
}

impl ops::Sub<Scalar> for Scalar {
    type Output = Scalar;

    fn sub(self, rhs: Scalar) -> Self::Output {
        let new_rhs = Forward::unary(Neg {}, rhs);
        Forward::binary(Add {}, self, new_rhs)
    }
}

impl ops::Neg for Scalar {
    type Output = Scalar;

    fn neg(self) -> Self::Output {
        Forward::unary(Neg {}, self)
    }
}

use proptest::prelude::*;

proptest! {
    #[test]
    fn add_test(a in any::<f64>(), b in any::<f64>()) {
        let res = Scalar::new(a) + Scalar::new(b);
        assert_eq!(a + b, res.v);
    }

    #[test]
    fn mul_test(a in any::<f64>(), b in any::<f64>()) {
        let res = Scalar::new(a) * Scalar::new(b);
        assert_eq!(a * b, res.v);
    }

    #[test]
    fn div_test(a in any::<f64>(), b in any::<f64>()) {
        let res = Scalar::new(a) / Scalar::new(b);
        if b == 0. {
            assert_eq!(0., res.v);
        } else {
            assert_eq!(a / b, res.v);
        }
    }

    #[test]
    fn sub_test(a in any::<f64>(), b in any::<f64>()) {
        let res = Scalar::new(a) - Scalar::new(b);
        assert_eq!(a - b, res.v);
    }

    #[test]
    fn neg_test(a in any::<f64>()) {
        let res = - Scalar::new(a);
        assert_eq!(-a, res.v);
    }

    #[test]
    fn ln_test(a in any::<f64>()) {
        let res = Scalar::new(a).ln();
        if a <= 0. {
            assert_eq!(0., res.v);
        } else {
            assert_eq!(a.ln(), res.v);
        }
    }

    #[test]
    fn exp_test(a in any::<f64>()) {
        let res = Scalar::new(a).exp();
        assert_eq!(a.exp(), res.v);
    }

    #[test]
    fn sig_test(a in any::<f64>()) {
        let res = Scalar::new(a).sig();
        if a >= 0. {
            assert_eq!(1. / (1. + (-a).exp()), res.v);
        } else {
            assert_eq!(a.exp() / (1. + a.exp()), res.v);
        }
    }

    #[test]
    fn relu_test(a in any::<f64>()) {
        let res = Scalar::new(a).relu();
        assert_eq!(a.max(0.), res.v);
    }

    #[test]
    fn eq_test(a in any::<f64>(), b in any::<f64>()) {
        let res = Scalar::new(a).eq(Scalar::new(b));
        if a == b {
            assert_eq!(1., res.v);
        } else {
            assert_eq!(0., res.v);
        }
    }

    #[test]
    fn lt_test(a in any::<f64>(), b in any::<f64>()) {
        let res = Scalar::new(a).lt(Scalar::new(b));
        if a < b {
            assert_eq!(1., res.v);
        } else {
            assert_eq!(0., res.v);
        }
    }

    #[test]
    fn gt_test(a in any::<f64>(), b in any::<f64>()) {
        let res = Scalar::new(a).gt(Scalar::new(b));
        if a > b {
            assert_eq!(1., res.v);
        } else {
            assert_eq!(0., res.v);
        }
    }
}
