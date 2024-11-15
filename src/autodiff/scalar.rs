use std::ops;

use rand::{thread_rng, Rng};

use super::{scalar_function::{Add, Inv, Log, Mul, ScalarFunction, Unary}, scalar_history::ScalarHistory};
use crate::autodiff::scalar_function::Binary;

// TODO: abstract over f64
#[derive(Debug)]
pub struct Scalar {
    v: f64,
    derivative: Option<f64>,
    history: ScalarHistory,
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

    pub fn log(self) -> Self {
        Forward::unary(Log {}, self)
    }
}

struct Forward;
impl Forward {
    fn binary(b: impl Binary + 'static, lhs: Scalar, rhs: Scalar) -> Scalar {
        let res = b.forward(&lhs.history.ctx, lhs.v, rhs.v);
        let new_history = ScalarHistory::default()
            .last_fn(ScalarFunction::B(Box::new(b)))
            .push_input(lhs)
            .push_input(rhs);
        Scalar::new(res).history(new_history)
    }
    
    fn unary(u: impl Unary + 'static, s: Scalar) -> Scalar {
        let res = u.forward(&s.history.ctx, s.v);
        let new_history = ScalarHistory::default()
            .last_fn(ScalarFunction::U(Box::new(u)))
            .push_input(s);
        Scalar::new(res).history(new_history)
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
        let denom = Forward::unary(Inv {}, rhs);
        Forward::binary(Mul {}, self, denom)
    }
}