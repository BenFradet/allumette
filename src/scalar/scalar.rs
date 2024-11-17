use std::ops;

use rand::{thread_rng, Rng};

use crate::{
    autodiff::forward::Forward,
    ops::{
        binary_ops::{Add, Div, Eq, Lt, Mul},
        unary_ops::{Exp, Ln, Neg, Relu, Sig},
    }, variable::Variable,
};

use super::{scalar_function::ScalarFunction, scalar_history::ScalarHistory};

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

impl Variable for Scalar {
    fn accumulate_derivative(mut self, x: f64) -> Self {
        if self.is_leaf() {
            self.derivative = Some(self.derivative.map(|d| d + x).unwrap_or(0.));
            self
        } else {
            self
        }
    }

    fn chain_rule(&self, d: f64) -> impl Iterator<Item = (&Self, f64)> {
        let derivatives = self.history.last_fn.as_ref()
            .map(|f| match f {
                ScalarFunction::B(b) => {
                    let (da, db) = b.backward(&self.history.ctx, d);
                    vec![da, db]
                },
                ScalarFunction::U(u) => {
                    let da = u.backward(&self.history.ctx, d);
                    vec![da]
                },
            })
            .unwrap_or(vec![]);
        let inputs = &self.history.inputs;
        inputs
            .iter()
            .zip(derivatives)
            //.filter(|(scalar, _)| !scalar.is_constant())
    }

    fn id(&self) -> u64 {
        self.id
    }

    fn is_constant(&self) -> bool {
        self.history.is_empty()
    }

    fn is_leaf(&self) -> bool {
        self.history.last_fn.is_none()
    }

    fn parents(&self) -> impl Iterator<Item = &Self> {
        self.history.inputs.iter()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{autodiff::context::Context, ops::binary_ops::Binary, scalar::{scalar_function::ScalarFunction, scalar_history::ScalarHistory}};

    struct F1;
    impl Binary for F1 {
        fn forward(&self, a: f64, b: f64) -> f64 {
            a + b + 10.
        }

        fn backward(&self, _ctx: &Context, d: f64) -> (f64, f64) {
            (d, d)
        }
    }

    struct F2;
    impl Binary for F2 {
        fn forward(&self, a: f64, b: f64) -> f64 {
            a * b + a
        }

        fn backward(&self, ctx: &Context, d: f64) -> (f64, f64) {
            let vs = &ctx.saved_values;
            let a = vs.first().unwrap_or(&1.);
            let b = vs.get(1).unwrap_or(&1.);
            (d * (b + 1.), d * a)
        }
    }

    #[test]
    fn chain_rule_test1() -> () {
        let hist = ScalarHistory::default()
            .last_fn(ScalarFunction::B(Box::new(F1 {})))
            .push_input(Scalar::new(0.))
            .push_input(Scalar::new(0.));
        let constant = Scalar::new(0.).history(hist);
        let back: Vec<_> = constant.chain_rule(5.).collect();
        assert_eq!(2, back.len());
        if let Some((_, deriv)) = back.first() {
            assert_eq!(5., *deriv);
        } else {
            panic!("test failure")
        }
    }

    #[test]
    fn chain_rule_test2() -> () {
        let f2 = F2 {};
        let y = Forward::binary(f2, Scalar::new(10.), Scalar::new(5.));
        let back: Vec<_> = y.chain_rule(5.).collect();
        assert_eq!(2, back.len());
        if let Some((_, deriv)) = back.get(1) {
            assert_eq!(50., *deriv);
        } else {
            panic!("test failure")
        }
    }

    #[test]
    fn chain_rule_test3() -> () {
        let f2 = F2 {};
        let v1 = Scalar::new(5.);
        let v1_id = v1.id;
        let v2 = Scalar::new(10.);
        let v2_id = v2.id;
        let y = Forward::binary(f2, v1, v2);
        let back: Vec<_> = y.chain_rule(5.).collect();
        assert_eq!(2, back.len());
        if let Some((v, deriv)) = back.first() {
            assert_eq!(v1_id, v.id);
            assert_eq!(55., *deriv);
        } else {
            panic!("test failure")
        }
        if let Some((v, deriv)) = back.get(1) {
            assert_eq!(v2_id, v.id);
            assert_eq!(25., *deriv);
        } else {
            panic!("test failure")
        }
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
