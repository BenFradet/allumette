use std::ops;

use rand::{thread_rng, Rng};

use super::{
    forward::Forward,
    scalar_function::{Add, Eq, Exp, Inv, Log, Lt, Mul, Neg, Relu, Sig},
    scalar_history::ScalarHistory,
};

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

    pub fn log(self) -> Self {
        Forward::unary(Log {}, self)
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
        let denom = Forward::unary(Inv {}, rhs);
        Forward::binary(Mul {}, self, denom)
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
