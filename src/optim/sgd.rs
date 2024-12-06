use crate::{module::parameter::Parameter, scalar::scalar::Scalar};

use super::optimizer::Optimizer;

pub struct SGD {
    params: Vec<Parameter>,
    lr: f64,
}

impl SGD {
    pub fn new(params: Vec<Parameter>, lr: f64) -> Self {
        Self {
            params,
            lr,
        }
    }
}

impl Default for SGD {
    fn default() -> Self {
        Self {
            params: vec![],
            lr: 1.,
        }
    }
}

impl Optimizer for SGD {
    fn zero(mut self) -> Self {
        for p in &mut self.params {
            p.scalar.derivative = None;
        }
        self
    }

    fn step(mut self) -> Self {
        for p in &mut self.params {
            match p.scalar.derivative {
                Some(d) => {
                    p.scalar = Scalar::new(p.scalar.v - self.lr * d);
                },
                None => (),
            }
            p.scalar.derivative = None;
        }
        self
    }
}