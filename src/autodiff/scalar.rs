use std::ops;

use rand::{thread_rng, Rng};

use super::{scalar_function::{Add, ScalarFunction}, scalar_history::ScalarHistory};
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
}

impl ops::Add<Scalar> for Scalar {
    type Output = Scalar;

    fn add(self, rhs: Scalar) -> Self::Output {
        let a = Add {};
        Forward::binary(a, self, rhs)
    }
}