use std::collections::HashMap;

use crate::scalar::scalar::Scalar;

use super::optimizer::Optimizer;

pub struct SGD {
    lr: f64,
}

impl SGD {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl Optimizer for SGD {
    fn zero(&self, mut scalars: HashMap<String, Scalar>) -> HashMap<String, Scalar> {
        for s in scalars.values_mut() {
            s.derivative = None;
        }
        scalars
    }

    fn step(&self, mut scalars: HashMap<String, Scalar>) -> HashMap<String, Scalar> {
        for s in scalars.values_mut() {
            if let Some(d) = s.derivative {
                *s = Scalar::new(s.v - self.lr * d).id(s.id.clone());
            }
            s.derivative = None;
        }
        scalars
    }
}
